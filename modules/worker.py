import os
import gc
import cv2
import json
import torch
import time
import logging
logging.getLogger("atom").setLevel(logging.INFO)
logging.getLogger("bytedkafka").setLevel(logging.WARNING)
logging.getLogger("kafka").setLevel(logging.WARNING)

from functools import partial
from threading import Semaphore
from traceback import format_exc
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from bytedabase import Client as AbaseClient
from bytedkafka import BytedKafkaConsumer

from atom.utils.logger import logger
from atom.modules.vector_retrieval.modules.vikingdb import VikingDBManager
from atom.modules.vector_retrieval.training.modules import VideoProcessor

from atom.modules.vector_retrieval.modules.utils import (
    now, 
    shots_stamp,
    arithmetic_stamp,
    shot_segmentation,
    legal_shot_cache,
    legal_task_status, 
    legal_kafka_message,
)
from atom.modules.drama_storyline_creative_generation.utils.cutting_utils import download_video


class Worker:
    def __init__(
        self,
        abase_config:dict,
        kafka_config:dict,
        vikingdb_config:dict,
        video_processor:VideoProcessor,
        concurrent:int=1,
    ):
        self.abase_config = abase_config
        self.kafka_config = kafka_config
        self.vikingdb_config = vikingdb_config

        self.concurrent = concurrent
        self.semaphore = Semaphore(concurrent)
        self.process_executor = ProcessPoolExecutor(max_workers=concurrent)
        self.thread_executor = ThreadPoolExecutor(max_workers=concurrent)

        self.video_processor = video_processor.eval()

        # lazy initialization
        self.vikingdb_manager = None
        self.abase_client = None
        self.kafka_consumer = None

    def __reduce__(self):
        return (
            self.__class__, 
            (
                self.abase_config, 
                self.kafka_config, 
                self.vikingdb_config, 
                self.video_processor, 
                self.concurrent,
            ),
        )
    
    def init_kafka(self):
        logger.info("初始化Kafka")
        self.kafka_consumer = BytedKafkaConsumer(**self.kafka_config)

    def init_abase(self):
        logger.info("初始化Abase")
        self.abase_client = AbaseClient(**self.abase_config)

    def init_vikingdb(self):
        logger.info("初始化VikingDB")
        self.vikingdb_manager = VikingDBManager(**self.vikingdb_config)
        
    @torch.no_grad()
    def extract_features(self, frames:list, batch_size=16)->list:
        features = []
        for batch in range(0, len(frames), batch_size):
            batch_frames = frames[batch:batch+batch_size]
            batch_frames = self.video_processor.transform_frames(batch_frames) # [B, C, H, W]
            feature = self.video_processor.extract_features(batch_frames) # [B, D]
            features.append(feature.detach().cpu())
        if len(features) == 0:
            return []
        return torch.cat(features, dim=0).tolist()

        # frames = self.video_processor.transform_frames(frames) # [B, C, H, W]
        # features = self.video_processor.extract_features(frames) # [B, D]
        # features = features.detach().cpu()
        # return features.tolist()

    def extract_video(self, req:dict):
        task_id = req["task_id"]
        vid = req["video_id"]
        video_path = req.get("video_path", f"video_cache/{vid}.mp4") # 兼容本地调试，线上环境统一用vid表示
        shot_segment = req["shot_segment"]
        max_interval = req["max_interval"]
        interval = req["interval"]

        if shot_segment:
            data = self.abase_client.get(vid) # 尝试获取镜头分割的缓存
            if legal_shot_cache(data):
                logger.info(f"读取{vid}镜头分割缓存")
                value = json.loads(data)
                shots, fps = value["shots"], value["fps"]
            else:
                logger.info(f"{vid}开始提取帧戳")
                data = self.abase_client.get(task_id)
                if not legal_task_status(data):
                    logger.info(f"镜头分割时发现任务{task_id}的缓存数据不合法:{data}，强制覆写")
                    value = {"status": "processing", "create_time": now()}
                else:
                    value = json.loads(data)
                value.update({"status": "segmenting", "update_time": now()})
                value_str = json.dumps(value, ensure_ascii=False)
                self.abase_client.set(task_id, value_str)

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"无法打开视频文件{video_path}")
                fps = round(cap.get(cv2.CAP_PROP_FPS))
                segmentation_time_start = time.perf_counter()
                value = shot_segmentation(vid)
                segmentation_time_end = time.perf_counter()
                segmentation_time_elapsed = segmentation_time_end - segmentation_time_start
                logger.info(f"任务 {task_id} 镜头分割执行时间 {segmentation_time_elapsed:.2f}s")
                shots, server_fps = value["shots"], value["fps"]
                assert abs(fps-server_fps) < 1, f"本地视频帧率{fps}与镜头分割服务帧率{server_fps}不同"

                value_str = json.dumps(value, ensure_ascii=False)
                self.abase_client.set(vid, value_str)
                logger.info(f"{vid}写入镜头分割缓存")

            framestamps, timestamps = shots_stamp(shots, fps, max_interval)
        else: # 采用等距采样
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件{video_path}")
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            framestamps, timestamps = arithmetic_stamp(total_frames, fps, interval)

        logger.info(f"{vid}帧分割完毕")
        data = self.abase_client.get(task_id)
        if not legal_task_status(data):
            logger.info(f"抽帧时发现任务{task_id}的缓存数据不合法:{data}，强制覆写")
            value = {"status": "processing", "create_time": now()}
        else:
            value = json.loads(data)
        value.update({"status": "extracting", "update_time": now()})
        value_str = json.dumps(value, ensure_ascii=False)
        self.abase_client.set(task_id, value_str)

        frames = self.video_processor.extract_frame(video_path, framestamps)
        logger.info(f"{vid}共提取{len(frames)}帧")
        # future = self.process_executor.submit(
        #     self.extract_features,
        #     frames=frames,
        # )
        # features = future.result()
        start_time = time.perf_counter()
        features = self.extract_features(frames=frames)
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time)
        logger.info(f"任务 {task_id} 特征提取完毕, 用时{inference_time_ms:.2f}s")
        return features, timestamps

    def callback(self, future:Future, req:dict):
        task_id = req["task_id"]
        video_id = req["video_id"]
        data = self.abase_client.get(task_id)
        if not legal_task_status(data):
            logger.info(f"任务结束时发现任务{task_id}的缓存数据不合法:{data}，强制覆写")
            value = {"status": "processing", "create_time": now()}
        else:
            value = json.loads(data)

        if future.exception():
            logger.info(f"任务{task_id}执行失败，错误信息为:\n{format_exc()}")
            value.update({"status": "fail", "result": str(future.exception()), "update_time": now()})
        else:
            success, result = future.result()
            status = "success" if success else "fail"
            value.update({"status": status, "result": result, "update_time": now()})

        value_str = json.dumps(value, ensure_ascii=False)
        self.abase_client.set(task_id, value_str)
        self.semaphore.release()
        
        if os.path.exists(f"video_cache/{video_id}.mp4"):
            os.remove(f"video_cache/{video_id}.mp4")
        logger.info(f"任务{task_id}处理完毕")
        # total_time_start = req["total_time_start"]
        # total_time_end = time.perf_counter()
        # total_time_elapsed = total_time_end - total_time_start
        # logger.info(f"任务 {task_id} 执行总时间 {total_time_elapsed:.2f}s")
        # logger.info(f"----------任务 {task_id} 结束-----------")
        # print()
        gc.collect()

    def process(self, req:dict):
        task_type = req["task_type"]
        video_id = req["video_id"]
        source = req["source"]

        features, timestamps = self.extract_video(req)

        if self.vikingdb_manager is None:
            self.init_vikingdb()

        if task_type == "upload":
            book_id = req["book_id"]
            episode_id = req["episode_id"]
            success, result = self.vikingdb_manager.upload_vector(
                vectors=features,
                timestamps=timestamps,
                book_id=book_id,
                episode_id=episode_id,
                video_id=video_id,
                source=source,
            )
            if success:
                result = len(result) # 记录该视频上传了多少个特征
            return success, result
        
        if task_type == "recall":
            book_id = req.get("book_id", None)
            source = req.get("source", None)
            topk = req["topk"]
            index = req["index"]
            success, result = self.vikingdb_manager.recall_vector(
                vectors=features,
                book_id=book_id,
                source=source,
                topk=topk,
                index=index,
            )
            if success:
                # TODO 将时间戳整合成脚本格式
                temp = []
                for shot in result:
                    tmp = []
                    for topk in shot:
                        t = {}
                        t["bid"] = topk["book_id"]
                        t["epid"] = topk["episode_id"]
                        t["start"] = topk["timestamp"][0]
                        t["end"] = topk["timestamp"][1]
                        t["scores"] = topk["scores"]
                        tmp.append(t)
                    temp.append(tmp)
                result = temp
            return success, result

    def run(self):
        if self.kafka_consumer is None:
            self.init_kafka()
        while True:
            self.semaphore.acquire()
            msg = self.kafka_consumer.__next__()
            self.kafka_consumer.commit()
            if not legal_kafka_message(msg):
                logger.info(f"请求不合法:{msg}")
                self.semaphore.release()
                continue
            req = json.loads(msg.value)
            task_id = req["task_id"]
            video_id = req["video_id"]
            task_type = req["task_type"]
            total_time_start = time.perf_counter()
            logger.info(f"----------任务 {task_id} 开始-----------")
            logger.info(f"收到任务{task_id}, vid为{video_id}, 任务类型为{task_type}")
            if self.abase_client is None:
                self.init_abase()
            data = self.abase_client.get(task_id)
            if not legal_task_status(data):
                logger.info(f"任务{task_id}的缓存数据不合法:{data}") # 这里主要考虑的是kafka producer未能正确将任务状态写入的情况
                self.semaphore.release()
                continue
            value = json.loads(data)
            value.update({"status": "processing", "update_time": now()})
            value_str = json.dumps(value, ensure_ascii=False)
            self.abase_client.set(task_id, value_str)
            
            if task_type == "upload": # 短剧原片
                base_url = "https://usergrowth.com.cn/open_api/video?video_id="
            if task_type == "recall": # 素材
                base_url = "https://5nv4loz5.fn.bytedance.net/video?video_id="
            try:
                download_time_start = time.perf_counter()
                if not os.path.exists(f"video_cache/{video_id}.mp4"):
                    download_video(video_id, base_url=base_url)
                download_time_end = time.perf_counter()
                download_time_elapsed = download_time_end - download_time_start
                logger.info(f"任务 {task_id} 下载视频总时间 {download_time_elapsed:.2f}s")
            except Exception:
                logger.info(f"{video_id}下载失败:\n{format_exc()}")
                value.update({"status": "fail", "result": f"{video_id}下载失败", "update_time": now()})
                value_str = json.dumps(value, ensure_ascii=False)
                self.abase_client.set(task_id, value_str)
                self.semaphore.release()
                continue
            # 执行任务
            req["total_time_start"] = total_time_start
            try:
                future = self.thread_executor.submit(self.process, req)
                future.add_done_callback(partial(self.callback, req=req))
                future.result()
            except Exception as e:
                logger.error(f"任务 {req.task_id} 回调出错: {str(e)}")
            finally:
                total_time_start = req["total_time_start"]
                total_time_end = time.perf_counter()
                total_time_elapsed = total_time_end - total_time_start
                logger.info(f"任务 {task_id} 执行总时间 {total_time_elapsed:.2f}s")
                logger.info(f"----------任务 {task_id} 结束-----------")
                print()


            

    def test_gpu_run(self):
        test_bid = "7520442672071314494"
        video_id = "v02ebeg10000d1gf55fog65neom5i2q0" # ep0
        task_type = "upload"
        task_id = "upload_"+video_id

        req = {
            "task_type": task_type,
            "video_id": video_id,
            "source": "fanqie",
            "task_id": task_id,
            "shot_segment": True,
            "max_interval": 2.0,
            "use_cache": False,
            "interval": 1.0,
        }
        base_url = "https://usergrowth.com.cn/open_api/video?video_id="
        try:
            if not os.path.exists(f"video_cache/{video_id}.mp4"):
                download_video(video_id, base_url=base_url)
        except Exception:
            logger.info(f"{video_id}下载失败:\n{format_exc()}")
            # value.update({"status": "fail", "result": f"{video_id}下载失败", "update_time": now()})
            # value_str = json.dumps(value, ensure_ascii=False)
            # self.abase_client.set(task_id, value_str)
            # self.semaphore.release()
        # 执行任务
        self.process(req)

    def __call__(self):
        try:
            self.run()
        except Exception:
            logger.info(f"worker出现异常:\n{format_exc()}")
            logger.info(f"重新启动中...")
        except KeyboardInterrupt:
            pass
        except SystemExit:
            pass
        except GeneratorExit:
            pass
        finally:
            if self.kafka_consumer is not None:
                self.kafka_consumer.close()
            self.process_executor.shutdown()
            self.thread_executor.shutdown()


import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("--concurrent", "-cc", help="并发数", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    concurrent = args.concurrent
    # concurrent = 1
    abase_config = {
        "psm":"abase_dm_tc21_feature",
        "table":"base_algo",
    }
    kafka_config = {
        "cluster":"bmq_hotsoon",
        "topic":"flower_task_ploy_abase_dump_topic",
        "group_id":"drama_group",
        "prefer_service": "dcleader",
        "ignore_dc_check": "false",
        # "max_poll_interval_ms": 1000*60*10,
        # "session_timeout_ms": 1000*60*10,
        "max_poll_records": 1,
        "stand_by": True,
        "enable_auto_commit": False,
    }
    vikingdb_config = {
        "vikingdb_name":"user_growth_1748420563__drama",
        "token":"ff0b99583c1716d24b358355a9800fdf",
        "region":"CN",
    }
    device = "cuda"
    video_processor = torch.load("atom/modules/vector_retrieval/model.pth", weights_only=False, map_location=device)
    video_processor.device = device
    worker = Worker(
        abase_config=abase_config,
        kafka_config=kafka_config,
        vikingdb_config=vikingdb_config,
        video_processor=video_processor,
        concurrent=concurrent,
    )
    worker()
    # import time
    # req = {
    #     "video_id": None,
    #     "video_path": "/mlx_devbox/users/wangxiangfeng/playground/atom/video_cache/raw.mp4",
    #     "shot_segment": False,
    # }
    # t0 = time.time()
    # futures = []
    # for i in range(4):
    #     future = worker.thread_executor.submit(worker.extract_video, req)
    #     futures.append(future)
    # worker.thread_executor.shutdown()
    # t1 = time.time()
    # print(t1-t0)
