import euler
from euler.base_compat_middleware import server_middleware
euler.install_thrift_import_hook()

import os
import json
import torch
from traceback import format_exc
from argparse import ArgumentParser
from multiprocessing import Process

from bytedkafka import BytedKafkaProducer
from bytedabase import Client as AbaseClient

from atom.utils.logger import logger
from atom.modules.vector_retrieval.modules.worker import Worker
from atom.modules.vector_retrieval.training.modules import VideoProcessor
from atom.modules.vector_retrieval.modules.utils import now, legal_task_status

from atom.idls.dvr.retrieval_thrift import (
    SubmitRequest,
    SubmitResponse,
    QueryRequest,
    QueryResponse,
    DramaVideoRetrievalService,
)

server = euler.Server(DramaVideoRetrievalService)
server.use(server_middleware)

@server.register("submit_task")
def submit_task(ctx, req: SubmitRequest)->SubmitResponse:
    logger.info(f"Received request: {req}")
    video_id = req.video_id
    task_type = req.task_type
    extra_info = req.extra_info
    # 首先对请求进行格式校验
    try:
        extra_info = json.loads(extra_info)
    except Exception:
        return SubmitResponse(status_code=1, task_id="", msg="extra_info不合法")
    
    use_cache = extra_info.get("use_cache", True)
    if task_type == "upload":
        if "episode_id" not in extra_info:
            return SubmitResponse(status_code=1, task_id="", msg="上传任务缺少episode_id")
        if "book_id" not in extra_info:
            return SubmitResponse(status_code=1, task_id="", msg="上传任务缺少book_id")
        if "source" not in extra_info:
            return SubmitResponse(status_code=1, task_id="", msg="上传任务缺少source")
        kafka_message = {
            "task_id": f"upload_{video_id}",
            "video_id": video_id,
            "book_id": extra_info["book_id"],
            "episode_id": extra_info["episode_id"],
            "source": extra_info["source"],
            "task_type": "upload",
            "shot_segment": extra_info.get("shot_segment", True),
            "max_interval": extra_info.get("max_interval", 2.0),
            "interval": extra_info.get("interval", 1.0),
        }
    elif task_type == "recall":
        kafka_message = {
            "task_id": f"recall_{video_id}",
            "video_id": video_id,
            "task_type": "recall",
            "source": extra_info.get("source", None),
            "book_id": extra_info.get("book_id", None),
            "index": extra_info.get("index", "bruteforce"),
            "topk": extra_info.get("topk", 3),
            "shot_segment": extra_info.get("shot_segment", True),
            "max_interval": extra_info.get("max_interval", 2.0),
            "interval": extra_info.get("interval", 1.0),
        }
    else:
        return SubmitResponse(status_code=1, task_id="", msg="task_type不合法")
    
    abase_client = AbaseClient(
        psm="abase_dm_tc21_feature",
        table="base_algo",
    )
    if use_cache:
        data = abase_client.get(kafka_message["task_id"])
        if legal_task_status(data):
            value = json.loads(data)
            if value["status"] != "fail": # 如果fail就重新提交
                return SubmitResponse(
                    status_code=0, 
                    task_id=kafka_message["task_id"], 
                    msg="duplicate",
                    extra_info=json.dumps({
                        "create_time": value["create_time"],
                    }, ensure_ascii=False)
                )

    # 写入Abase
    value = {"status": "pending", "video_id": video_id, "create_time": now()}
    value_str = json.dumps(value, ensure_ascii=False)
    abase_client.set(kafka_message["task_id"], value_str)
        
    # 发送kafka消息
    topic = "flower_task_ploy_abase_dump_topic"
    producer = BytedKafkaProducer(
        cluster="bmq_hotsoon",
        topics=[topic],
        group_id="drama_group",
        # group_id="drama_group_test",
        linger_ms=1000,
    )
    future = producer.send(topic, json.dumps(kafka_message, ensure_ascii=False).encode())
    try:
        future.get()
    except Exception as e:
        return SubmitResponse(
            status_code=1,
            task_id=kafka_message["task_id"],
            msg=str(e),
        )

    logger.info("kafka消息发送成功")
    producer.flush()
    producer.close()
    
    return SubmitResponse(
        status_code=0, 
        task_id=kafka_message["task_id"], 
        msg="success",
        extra_info=json.dumps({"create_time": value["create_time"]}, ensure_ascii=False),
    )
    
@server.register("query_task")
def query_task(ctx, req:QueryRequest)->QueryResponse:
    logger.info(f"Received request: {req}")
    task_id = req.task_id
    abase_client = AbaseClient(
        psm="abase_dm_tc21_feature",
        table="base_algo",
    )
    data = abase_client.get(task_id)
    if not legal_task_status(data):
        return QueryResponse(status_code=1, task_id=task_id, task_status="task_id不存在或不合法")
    value = json.loads(data)
    return QueryResponse(
        status_code=0, 
        task_id=task_id,
        task_status=value.pop("status"), 
        extra_info=json.dumps(value, ensure_ascii=False)
    )

def start_worker():
    concurrent = 1
    abase_config = {
        "psm":"abase_dm_tc21_feature",
        "table":"base_algo",
    }
    kafka_config = {
        "cluster":"bmq_hotsoon",
        "topic":"flower_task_ploy_abase_dump_topic",
        "group_id":"drama_group_1",
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
    video_processor:VideoProcessor = torch.load("atom/modules/vector_retrieval/model.pth", weights_only=False, map_location="cpu")
    while True:
        try:
            worker = Worker(
                abase_config=abase_config,
                kafka_config=kafka_config,
                vikingdb_config=vikingdb_config,
                video_processor=video_processor,
                concurrent=concurrent,
            )
            worker()
        except:
            break

if __name__ == "__main__":
    parser = ArgumentParser("Drama Video Retrieval Service")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    # 启动worker
    logger.info("Starting worker...")
    p = Process(target=start_worker)
    p.start()

    # 启动tce服务
    try:
        logger.info("Starting server...")
        server.run('tcp://[::]:{}'.format(args.port), workers_count=args.workers)
    except:
        logger.error(format_exc())
    finally:
        p.kill()
        server.stop()
        logger.info("Server exiting")
