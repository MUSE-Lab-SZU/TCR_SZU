import json
import time
from bytedkafka import BytedKafkaProducer
from bytedabase import Client as AbaseClient
from atom.modules.vector_retrieval.modules.utils import now

if __name__ == "__main__":
    topic = "flower_task_ploy_abase_dump_topic"
    kafka_config = {
        "cluster": "bmq_hotsoon",
        "topics": [topic],
        "group_id": "drama_group_test",
        'linger_ms': 1000,
    }
    producer = BytedKafkaProducer(**kafka_config)
    abase_config = {
        "psm":"abase_dm_tc21_feature",
        "table":"base_algo",
    }
    abase_client = AbaseClient(**abase_config)

    messages = [
    {
        "task_id": "upload_v0debeg10000cskon97og65h132ulang",
        "video_id": "v0debeg10000d0r7pofog65mbik24jf0",
        "book_id": "7433359455607082046",
        "episode_id": 7,
        "source": "fanqie",
        "task_type": "upload",
        "shot_segment": True,
        "max_interval": 2.0,
        "interval": 1.0,
    },
    # {
    #     # "task_id": "upload_v0debeg10000d0r7povog65mrupf8ql0",
    #     "task_id": "recall_v0debeg10000d0r7povog65mrupf8ql0",
    #     "video_id": "v0debeg10000d0r7povog65mrupf8ql0",
    #     "book_id": "7509313162869492761",
    #     "episode_id": 5,
    #     "source": "fanqie",
    #     "shot_segment": True,
    #     # "task_type": "upload",
    #     "task_type": "recall",
    #     "index": "bruteforce",
    #     "topk": 3,
    # },
    ]
    for msg in messages:
        value = {"status": "pending", "video_id": msg["video_id"], "create_time": now()}
        value_str = json.dumps(value, ensure_ascii=False)
        abase_client.set(msg["task_id"], value_str)
        future = producer.send(topic, json.dumps(msg).encode())

    producer.flush()
    producer.close()
    print("kafka消息发送成功")

    while True:
        print(abase_client.get("upload_v0debeg10000d0r7pofog65mbik24jf0").decode())
        # print(abase_client.get("recall_v0debeg10000d0r7povog65mrupf8ql0").decode())
        time.sleep(5)
