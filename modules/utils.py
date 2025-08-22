import euler
from euler.base_compat_middleware import client_middleware
euler.install_thrift_import_hook()

import json
import time
import heapq
from tqdm import tqdm
from datetime import datetime

from atom.modules.drama_storyline_creative_generation.utils.cutting_utils import get_bid_info

from atom.idls.dvr.retrieval_thrift import (
    SubmitRequest,
    SubmitResponse,
    QueryRequest,
    QueryResponse,
    DramaVideoRetrievalService,
)

from atom.idls.isp.multimodel_insights_inference_thrift import (
    MultimodelInsightsInferenceRequest,
    MultimodelInsightsInferenceResponse,
    MultimodelInsightsInferenceService,
)

isp_psm = "sd://isp.lmm.multimodel_insights_inference?cluster=default"
isp_client = euler.Client(MultimodelInsightsInferenceService, isp_psm, timeout=3000)
isp_client.use(client_middleware)

dvr_psm = "sd://toutiao.growth.dvr?cluster=default"
dvr_client = euler.Client(DramaVideoRetrievalService, dvr_psm, timeout=3000)
dvr_client.use(client_middleware)

def get_today_as_number():
    today = datetime.today()
    return int(today.strftime('%Y%m%d'))

def now():
    # 获取当前时间
    now = datetime.now()
    # 按指定格式格式化输出
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_time

def legal_kafka_message(msg):
    try:
        req = json.loads(msg.value)
    except Exception:
        return False
    if not isinstance(req, dict):
        return False
    if "task_id" not in req:
        return False
    if "video_id" not in req:
        return False
    if "source" not in req:
        return False
    if "task_type" not in req:
        return False
    if "shot_segment" not in req:
        return False
    if "max_interval" not in req:
        return False
    if "interval" not in req:
        return False
    if req["task_type"] == "upload":
        if "episode_id" not in req:
            return False
        if "book_id" not in req:
            return False
    elif req["task_type"] == "recall":
        if "index" not in req:
            return False
        if "topk" not in req:
            return False
    else:
        return False
    return True

def legal_task_status(data):
    try:
        value = json.loads(data)
    except Exception:
        return False
    if "status" not in value:
        return False
    if "video_id" not in value:
        return False
    if "create_time" not in value:
        return False
    if value["status"] not in ["pending", "processing", "segmenting", "extracting", "success", "fail"]:
        return False
    if value["status"] in ["success", "fail"] and "result" not in value:
        return False
    return True

def legal_shot_cache(data):
    try:
        value = json.loads(data)
    except Exception:
        return False
    if "shots" not in value:
        return False
    if not isinstance(value["shots"][0], list):
        return False
    if "fps" not in value:
        return False
    return True

def shot_segmentation(vid):
    meta_info = {"result_content": "frame"}
    meta_info = json.dumps(meta_info, ensure_ascii=False)

    req = MultimodelInsightsInferenceRequest(
        vid=vid,
        video_meta=meta_info,
        model="isp_algo_mllm_insights_shot_boundary_detect",
    )
    attempt = 0
    while attempt < 3:
        res = isp_client.MultimodelInsightsInference(request=req)
        result = json.loads(res.response)
        if len(result["open_label_list"]) > 0:
            shots_str = result["open_label_list"][0]
            fps = result["extra"]["fps"]
            shots = json.loads(shots_str)
            return {"shots": shots, "fps": fps}
        else:
            print(f"{vid}镜头分割未返回结果")
            attempt += 1
    raise ValueError(f"{vid}镜头分割失败")

def arithmetic_stamp(total_frame:int, fps:float, interval:float):
    frame_stamps, time_stamps = [], []
    present_frame = round(fps * interval / 2)
    left, right = 0, round(fps * interval)
    while present_frame <= total_frame:
        frame_stamps.append(present_frame)
        time_stamps.append([int(left/fps*1000), int(right/fps*1000)])
        present_frame += round(fps * interval)
        left += round(fps * interval)
        right += round(fps * interval)
        right = min(right, total_frame)
    return frame_stamps, time_stamps    

def shots_stamp(shots:list, fps:float, max_interval:float=2.0):
    frame_stamps, time_stamps = [], []
    for shot in shots:
        left, right = shot[0], shot[1]
        interval = right - left
        split = int(interval / fps / max_interval) + 1
        mini_interval = interval / (split+1)
        mini_frames = [round(left + mini_interval * (i+1)) for i in range(split)]
        mini_time = []
        if split > 1:
            for i in range(split):
                if i == 0:
                    mini_time.append([left, round(left+1.5*mini_interval)])
                elif i == split-1:
                    mini_time.append([round(right-1.5*mini_interval), right])
                else:
                    mini_time.append([round(left+mini_interval*(i+0.5)), round(left+mini_interval*(i+1.5))])
        else:
            mini_time.append(shot)
        mini_time = [[int(l/fps*1000), int(r/fps*1000)] for l,r in mini_time]
        frame_stamps.extend(mini_frames)
        time_stamps.extend(mini_time)
    return frame_stamps, time_stamps

def post_process(
    recall:list, 
    connect_threshold:float=2000,
    segment_threshold:float=3500,
):
    result = []
    chunk = {}
    consistency = 1
    for r in recall:
        bid = r["bid"]
        epid = r["epid"]
        start = r["start"]
        end = r["end"]
        if not chunk:
            chunk = {
                "bid": bid,
                "epid": epid,
                "start": start,
                "end": end,
            }
        elif chunk["bid"] != bid or chunk["epid"] != epid:
            if consistency >= 3 or chunk["end"]-chunk["start"] >= segment_threshold: # 单位毫秒
                result.append(chunk)
            consistency = 1
            chunk = {
                "bid": bid,
                "epid": epid,
                "start": start,
                "end": end,
            }
        elif abs(start-chunk["end"]) <= connect_threshold:
            consistency += 1
            if end > chunk["end"]:
                chunk["end"] = end
        else:
            if consistency >= 3 or chunk["end"]-chunk["start"] >= segment_threshold: # 单位毫秒
                result.append(chunk)
            consistency = 1
            chunk = {
                "bid": bid,
                "epid": epid,
                "start": start,
                "end": end,
            }
    if chunk:
        if consistency >= 3 or chunk["end"]-chunk["start"] >= segment_threshold: # 单位毫秒
            result.append(chunk)
    return result

def compute_distance(anchor, target):
    distance = 0
    if anchor["bid"] != target["bid"] or anchor["epid"] != target["epid"]:
        distance += 10_0000
    else:
        distance += abs(target["start"] - anchor["end"]) / 1000 # 单位是秒    
    return distance

def beam_search(recall:list, beam_size:int = 3):
    """
    Perform beam search to find the minimum-cost path through candidate segments.
    
    Args:
        paths (List[List[Dict]]): A list of length n, each element is a list of k candidate segments.
        beam_size (int): The beam size for the search.
    
    Returns:
        Tuple[List[Dict], float]: The best path (sequence of segments) and its total cost.
    """
    # Each beam is a tuple (cost, path), where cost is the accumulated cost
    # and path is a list of segments chosen so far.
    beams = [(0.0, [candidate]) for candidate in recall[0]]
    
    for t in range(1, len(recall)):
        next_beams = []
        for cost, path in beams:
            prev_seg = path[-1]
            for candidate in recall[t]:
                new_cost = cost + compute_distance(prev_seg, candidate)
                new_path = path + [candidate]
                next_beams.append((new_cost, new_path))
        # Keep only the beam_size best beams by cost
        beams = heapq.nsmallest(beam_size, next_beams, key=lambda x: x[0])
        if len(beams[0][1]) >= 2:
            best_cost, best_path = min(beams, key=lambda x: x[0])
            if best_cost >= 10_0000: # 出现了跨集或跨剧的情况，截断
                return best_path[:-1]
    # The first beam in the final list is the best one
    best_cost, best_path = min(beams, key=lambda x: x[0])
    return best_path

def filter_candidates(recall:list, beam_size:int=3):
    result = []
    index = 0
    while True:
        best_path = beam_search(recall[index:], beam_size)
        index += len(best_path)
        result.extend(best_path)
        if index == len(recall):
            return result

def upload_book(
    bid:str, 
    source:str="fanqie", 
    shot_segment:bool=True,
    max_interval:float=2.0,
    use_cache:bool=True
)->dict:
    bid_info = get_bid_info(bid, source)
    task_dict = {}
    total = len(bid_info["ep_to_vid"])
    pbar = tqdm(total=total, desc=f"{bid}提交任务进度")
    for epid, vid in bid_info["ep_to_vid"].items():
        req = SubmitRequest(
            video_id=vid,
            task_type="upload",
            extra_info=json.dumps({
                "book_id": bid,
                "episode_id": epid,
                "source": source,
                "shot_segment": shot_segment,
                "max_interval": max_interval,
                "use_cache": use_cache,
            }, ensure_ascii=False)
        )
        res:SubmitResponse = dvr_client.submit_task(req=req)
        task_dict[res.task_id] = (bid, epid, vid)
        pbar.update(1)
    pbar.close()
    return task_dict

def query_book(bid:str, task_list:list):
    total = len(task_list)
    pbar = tqdm(total=total, desc=f"{bid}任务完成进度")
    failed = 0
    total_frames = 0
    try:
        while True:
            unfinished = len(task_list)
            for _ in range(unfinished):
                task_id = task_list.pop(0)
                req = QueryRequest(
                    task_id=task_id,
                )
                res:QueryResponse = dvr_client.query_task(req=req)
                task_id = res.task_id
                if res.task_status == "success":
                    result = json.loads(res.extra_info)["result"]
                    total_frames += result
                    tqdm.write(f"{task_id}执行成功，共抽取{result}帧")
                    pbar.update(1)
                elif res.task_status == "fail":
                    result = json.loads(res.extra_info)["result"]
                    tqdm.write(f"{task_id}执行失败，原因为{result}")
                    failed += 1
                else:
                    task_list.append(task_id)
            if not task_list:
                tqdm.write(f"{bid}共抽取{total_frames}帧")
                pbar.close()
                return task_list
            time.sleep(5)
    except KeyboardInterrupt:
        pbar.close()
        return task_list

def upload_video(
    vid:str,
    bid:str,
    epid:int,
    source:str="fanqie",
    shot_segment:bool=True,
    max_interval:float=2.0,
    use_cache:bool=True,
    sync:bool=True,
):
    req = SubmitRequest(
        video_id=vid,
        task_type="upload",
        extra_info=json.dumps({
            "book_id": bid,
            "episode_id": epid,
            "source": source,
            "shot_segment": shot_segment,
            "max_interval": max_interval,
            "use_cache": use_cache,
        }, ensure_ascii=False)
    )
    res:SubmitResponse = dvr_client.submit_task(req=req)
    if not sync:
        return res.task_id
    req = QueryRequest(
        task_id=res.task_id,
    )
    while True:
        res:QueryResponse = dvr_client.query_task(req=req)
        if res.task_status in ["success", "fail"]:
            return res
        else:
            print(res)
        time.sleep(5)

def recall_video(
    vid:str, 
    bid:str=None, 
    source:str="fanqie", 
    topk:int=1, 
    index:str="bruteforce",
    shot_segment:bool=True,
    max_interval:float=2.0,
    use_cache:bool=True,
    sync:bool=True,
):
    if bid is None:
        print(f"[Warning] 召回{vid}时未传入book id，将从全部向量数据库范围检索，并不建议这么做")
    req = SubmitRequest(
        video_id=vid,
        task_type="recall",
        extra_info=json.dumps({
            "book_id": bid,
            "source": source,
            "topk": topk,
            "index": index,
            "shot_segment": shot_segment,
            "max_interval": max_interval,
            "use_cache": use_cache,
        }, ensure_ascii=False)
    )
    res:SubmitResponse = dvr_client.submit_task(req=req)
    if not sync:
        return res.task_id
    req = QueryRequest(
        task_id=res.task_id,
    )
    while True:
        res:QueryResponse = dvr_client.query_task(req=req)
        if res.task_status in ["success", "fail"]:
            return res
        else:
            print(res)
            time.sleep(5)
    
def display_frame(frames):
    import math
    from matplotlib import pyplot as plt
    num_cols = 8
    num_rows = math.ceil(len(frames) / num_cols)
    plt.figure(figsize=(num_cols*2, num_rows*2))

    for i, frame in enumerate(frames):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"{i} frame")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # print(arithmetic_stamp(100, 25, 1))
    # res = shot_segmentation("v0debeg10000cskonb7og65o88gh7sgg")
    shots = [[0, 38], [39, 102], [104, 220], [221, 295], [297, 390], [391, 452], [453, 487], [488, 536], [537, 713], [714, 807], [808, 1130], [1131, 1254], [1255, 1375], [1376, 1520], [1521, 1616], [1617, 1649], [1650, 1705], [1706, 1787], [1788, 1954], [1955, 2133], [2134, 2277], [2278, 2335], [2336, 2337], [2338, 2371], [2372, 2487], [2488, 2799], [2800, 2998], [2999, 3037], [3038, 3192], [3193, 3263]]
    fps = 25
    max_interval = 2
    frames, timestamps = shots_stamp(shots, fps, max_interval)
    print(frames)
    print(timestamps)
