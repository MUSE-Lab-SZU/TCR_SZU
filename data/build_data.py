import os
import cv2
import json
import random
import imageio
import numpy as np

from functools import partial
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
from torchvision import transforms

from bytedabase import Client as AbaseClient

from atom.utils.logger import logger
from atom.modules.vector_retrieval.training.modules import Resize
from atom.modules.drama_storyline_creative_generation.utils.mago_utils import get_dongcha_drama_list
from atom.modules.drama_storyline_creative_generation.utils.cutting_utils import get_bid_info, download_video
from atom.modules.vector_retrieval.modules.utils import (
    arithmetic_stamp, 
    shots_stamp, 
    shot_segmentation,
    legal_shot_cache,
)

def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

class DataBuilder:
    def __init__(
        self,  
        height:int=224,
        width:int=224,
        video_dir:str='/mnt/bn/vector2/videos',
        output_dir:str='/mnt/bn/vector2/tensors_320x180',
        concurrent:int=1,
        abase_config:dict=None,
    ):
        self.height = height
        self.width = width
        self.transform = transforms.Compose([
            Resize(height, width),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.concurrent = concurrent

        self.title_font = "atom/modules/vector_retrieval/assets/fonts/FZLTCHK.ttf"
        self.body_font = "atom/modules/vector_retrieval/assets/fonts/hanyiyakuhei-65.ttf"

        self.titles = ["免费看剧","观看全集","正在热播","下载观看"]

        fires = imageio.mimread("atom/modules/vector_retrieval/assets/images/fire.png")
        self.fires = [cv2.resize(fire, (101, 160)) for fire in fires]
        corner = cv2.imread("atom/modules/vector_retrieval/assets/images/corner.png", cv2.IMREAD_UNCHANGED)
        corner = cv2.cvtColor(corner, cv2.COLOR_BGRA2RGBA)
        self.corner = cv2.resize(corner, (150, 150))

        self.thread_executor = ThreadPoolExecutor(max_workers=concurrent)
        self.process_executor = ProcessPoolExecutor(max_workers=concurrent)

        self.abase_config = abase_config
        self.abase_client = None
        
    def __reduce__(self):
        return (
            self.__class__, 
            (
                self.height, 
                self.width, 
                self.video_dir, 
                self.output_dir,
                self.concurrent,
                self.abase_config,
            ),
        )
    
    def init_abase(self):
        logger.info("初始化Abase")
        self.abase_client = AbaseClient(**self.abase_config)

    def extract_frame(
        self, 
        video_path:str,
        framestamps:list,
    ):
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件{video_path}")
        for stamp in framestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, stamp)
            success, frame = cap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                logger.info(f"{os.path.basename(video_path)}无法读取帧{stamp}")
        cap.release()
        return frames
    
    def extract_framestamp(
        self, 
        video_path:str,
        vid:str=None, 
        shot_segment:bool=True, 
        interval:float=1.0,
    ):
        if shot_segment:
            assert vid is not None, "调用镜头分割必须传入vid"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件{video_path}")
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not shot_segment:
            framestamps, timestamps = arithmetic_stamp(total_frames, fps, interval)
        else:
            result = shot_segmentation(vid)
            shots, server_fps = result["shots"], result["fps"]
            assert abs(fps-server_fps) < 1, f"本地视频帧率{fps}与镜头分割服务帧率{server_fps}不同"
            framestamps, timestamps = shots_stamp(shots, fps)
        return framestamps, timestamps
    
    def overlay(self, background, front, position=None):
        bg = background.copy()
        logo = front.copy()
        if logo.shape[2] == 4:
            b, g, r, a = cv2.split(logo)
            logo_rgb = cv2.merge((b, g, r))
            alpha_mask = a / 255.0  # 归一化 alpha 掩码
        else:
            raise ValueError("Logo 没有 alpha 通道，请检查文件格式是否为 PNG 并带透明背景")
        if position is None:
            # 默认位置为背景图的右上角
            position = (0, bg.shape[1] - logo.shape[1])
        # 背景粘贴位置（右上角）
        x_offset, y_offset = position
        bg_roi = bg[x_offset:x_offset+logo.shape[0], y_offset:y_offset+logo.shape[1]]
        # 做 alpha blending
        for c in range(3):  # BGR 三通道
            bg_roi[..., c] = (alpha_mask * logo_rgb[..., c] + (1 - alpha_mask) * bg_roi[..., c]).astype('uint8')
        # 更新背景图
        bg[x_offset:x_offset+logo.shape[0], y_offset:y_offset+logo.shape[1]] = bg_roi
        return bg
    
    def render_frame(self, frames, drama_name):
        results = []
        if drama_name is None:
            drama_name = "番茄短剧"
        for f in frames:
            frame = f.copy()
            if frame.shape[:2] != (1920, 1080):
                frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_CUBIC)
            frame = self.overlay(frame, self.corner)
            frame = self.overlay(frame, random.choice(self.fires), position=(0, 320))
            frame = self.overlay(frame, random.choice(self.fires), position=(0, 660))
            frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame)

            text = "点\n击\n免\n费\n看\n全\n集\n*\n已\n完\n结\n*"
            font_size = 30
            font = ImageFont.truetype(self.body_font, font_size)
            position = (5, 660)
            color = "#ffffff"  # 白色
            draw.text(position, text, font=font, fill=color)

            text = "本故事纯属虚构"
            font_size = 40
            position = (540-len(text)*font_size/2, 1820-font_size/2)
            color = "#ffffff"  # 白色
            font = ImageFont.truetype(self.body_font, font_size)
            draw.text(position, text, font=font, fill=color)

            text = random.choice(self.titles)
            font_size = 55
            position = (540-len(text)*font_size/2, 100-font_size/2)
            color = random_color()
            font = ImageFont.truetype(self.body_font, font_size)
            draw.text(position, text, font=font, fill=color)

            text = drama_name
            font_size = 60
            position = (540-len(text)*font_size/2, 1536-font_size/2)
            color = random_color()
            font = ImageFont.truetype(self.title_font, font_size)
            draw.text(position, text, font=font, fill=color)

            results.append(np.array(frame))
        return results

    def frame_to_tensor(self, frames:list):
        tensors = []
        for frame in frames:
            frame = self.transform(frame)
            tensors.append(frame)
        return tensors

    def build_tensor(
        self,
        raw_video:str,
        save_file:str,
        vid:str=None,
        shot_segment:bool=True,
        max_interval:float=2.0,
        interval:float=1.0,
        drama_name:str=None,
        force_build:bool=False,
    ):
        if os.path.exists(save_file) and not force_build:
            try:
                content = torch.load(save_file)
                success = True
            except Exception:
                logger.info(f"{os.path.basename(save_file)}已损坏，开始修复")
                success = False
            if success:
                if "file" not in content:
                    content["file"] = os.path.basename(save_file)
                    torch.save(content, save_file)
                    logger.info(f"{os.path.basename(save_file)}增加文件名信息")
                    return content
                if "data" in content:
                    if len(content["data"]) == content["frame_num"]:
                        del content["data"]
                        logger.info(f"{os.path.basename(save_file)}已存在")
                        return content
                    else:
                        logger.info(f"{os.path.basename(save_file)}已损坏，开始修复")

        if shot_segment:
            data = self.abase_client.get(vid) # 尝试获取镜头分割的缓存
            if legal_shot_cache(data):
                logger.info(f"读取{vid}镜头分割缓存")
                value = json.loads(data)
                shots, fps = value["shots"], value["fps"]
            else:
                logger.info(f"{vid}开始提取帧戳")
                cap = cv2.VideoCapture(raw_video)
                if not cap.isOpened():
                    raise ValueError(f"无法打开视频文件{raw_video}")
                fps = round(cap.get(cv2.CAP_PROP_FPS))
                value = shot_segmentation(vid)
                shots, server_fps = value["shots"], value["fps"]
                assert abs(fps-server_fps) < 1, f"本地视频帧率{fps}与镜头分割服务帧率{server_fps}不同"
                value_str = json.dumps(value, ensure_ascii=False)
                self.abase_client.set(vid, value_str)
                logger.info(f"{vid}写入镜头分割缓存")
            framestamps, timestamps = shots_stamp(shots, fps, max_interval)
        else: # 采用等距采样
            cap = cv2.VideoCapture(raw_video)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件{raw_video}")
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            framestamps, timestamps = arithmetic_stamp(total_frames, fps, interval)

        frames = self.extract_frame(raw_video, framestamps)
        future = self.process_executor.submit(self.render_and_save, frames, drama_name)
        raw_tensors, render_tensors = future.result()
        content = {
            "file": os.path.basename(save_file),
            "data": [{"raw": raw, "render": render} for raw, render in zip(raw_tensors, render_tensors)],
            "frame_num": len(frames),
            "timestamps": timestamps,
        }
        torch.save(content, save_file)
        logger.info(f"{os.path.basename(save_file)}已保存")
        del content["data"]
        return content
    
    def render_and_save(self, frames, drama_name):
        render_frames = self.render_frame(frames, drama_name)
        raw_tensors = self.frame_to_tensor(frames)
        render_tensors = self.frame_to_tensor(render_frames)
        return raw_tensors, render_tensors
    
    def callback(self, future, meta_info):
        content = future.result()
        meta_info.append(content)

    def pipeline(
        self,
        book_ids:list, 
        max_episode:int, 
        video_dir:str=None,
        tensor_dir:str=None,
        force_download:bool=False,
        force_build:bool=False,
    ): 
        if video_dir is None:
            video_dir = self.video_dir
        if tensor_dir is None:
            tensor_dir = self.output_dir

        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(tensor_dir, exist_ok=True)

        if self.abase_client is None:
            self.init_abase()

        meta_info = []
        for bid in book_ids:
            info = get_bid_info(bid)
            ep_to_vid = info["ep_to_vid"]
            drama_name = info["drama_name"]
            logger.info(f"获取{bid}信息")
            for idx, (ep, vid) in enumerate(ep_to_vid.items()):
                file_name = f"{bid}_{str(ep).zfill(3)}"
                video_path = os.path.join(video_dir, file_name+".mp4")
                tensor_path = os.path.join(tensor_dir, file_name+".pt")
                if os.path.exists(video_path) and not force_download:
                    print(f"跳过下载{video_path}")
                else:
                    download_video(vid, output_file=video_path)
                kwds = {
                    "raw_video": video_path,
                    "save_file": tensor_path,
                    "vid": vid,
                    "shot_segment": True,
                    "max_interval": 2.0,
                    "interval": 1.0,
                    "drama_name": drama_name,
                    "force_build": force_build,
                }
                future = self.thread_executor.submit(self.build_tensor, **kwds)
                future.add_done_callback(partial(self.callback, meta_info=meta_info))
                if idx+1 == max_episode:
                    break

        self.thread_executor.shutdown()
        self.process_executor.shutdown()
        json.dump(meta_info, open(os.path.join(tensor_dir, "meta_info.json"), "w"))

if __name__ == "__main__":
    # bids = []
    # drama = get_dongcha_drama_list(begin_date="2025-04-19", end_date="2025-05-11")
    # for d in drama[::-1]:
    #     bid = d["book_id"]
    #     try:
    #         get_bid_info(bid)
    #         bids.append(bid)
    #         logger.info(f"{bid}的信息获取成功")
    #     except Exception:
    #         logger.info(f"未能获取{bid}的信息")
    #     if len(bids) == 50:
    #         break
    bids = """7491561174777990169
7493724179393432638
7493080842621717529
7494164417957858328
7493737557524499518
7493723524696132633
7490805472782470169
7490516727738944537
7492274919338691608
7495209339355925529
7494550535526157374
7494509549437586494
7494465888020597785
7491171885724486680
7487147674081184793
7491903551904418878
7493485960441629758
7494465824598543385
7494464178170678297
7494529695816371225
7495637639404457022
7495617429469400089
7495613226630007870
7495596646244305944
7494844217567480856
7494887930033343513
7494887226732465214
7494197984314264638
7495670455265676313
7495634677131119640
7495574585786518552
7495674236271332414
7495315021472222270
7494515571820596249
7495641983034084377
7494839365089102873
7494890329418517529
7496029372877376574
7496041320624950334
7491945949657320472
7495975268276784152
7495606593929612350
7494833683132599321
7493012760565926937
7496410757114907710
7495966377031781401
7495957858362608665
7495947804200684606
7494542047337008190
7495950587364658200""".split("\n")
    abase_config = {
        "psm":"abase_dm_tc21_feature",
        "table":"base_algo",
    }
    builder = DataBuilder(
        height=320,
        width=180,
        video_dir="/mnt/bn/vector2/videos",
        output_dir="/mnt/bn/vector2/tensors_320x180_shots",
        abase_config=abase_config,
        concurrent=8,
    )
    builder.pipeline(
        book_ids=bids, 
        max_episode=-1, 
        force_download=False, 
        force_build=False, 
    )
