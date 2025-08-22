import os
import time
import json
import torch

from viking.vikingdb_client import VikingDbData, VikingDbClient
from atom.modules.vector_retrieval.modules.utils import get_today_as_number

class VikingDBManager:
    def __init__(
        self, 
        vikingdb_name:str,
        token:str,
        region:str,
        **kwargs,
    ):
        self.client = VikingDbClient(
            vikingdb_name=vikingdb_name,
            token=token,
            region=region,
            **kwargs,
        )
    
    def upload_vector(
        self, 
        vectors:list, 
        timestamps:list,
        book_id:str, 
        episode_id:int, 
        video_id:str,
        source:str="fanqie",
    ):
        batch_data = []
        for idx, (vec, ts) in enumerate(zip(vectors, timestamps), start=1):
            dsl_info = {
                "book_id": book_id,
                "episode_id": episode_id,
                "timestamp": ts,
                "source": source,
                "video_id": video_id,
                "create_date": get_today_as_number(),
            }
            data_dict = {
                "label_upper64": int(book_id),
                "label_lower64": episode_id * 100000 + idx,
                "fvector": vec,
                "dsl_info": dsl_info,
            }
            data = VikingDbData(data_dict=data_dict)
            batch_data.append(data)
        err_msg, keys = self.client.simple_add_data(batch_data)
        if not err_msg:
            return True, keys
        else:
            return False, err_msg
        
    def recall_vector(
        self,
        vectors:list,
        book_id:str=None,
        source:str=None,
        topk:int=3,
        index:str="bruteforce",
        **kwargs,
    ):
        dsl_query = {}
        selector = {"op": "rowwise", "field": ["video_id", "book_id", "episode_id", "timestamp"],}
        dsl_query["selector"] = selector
        conditions = []
        if book_id is not None:
            conditions.append({"op": "must", "field": "book_id", "conds": [book_id]})
        if source is not None:
            conditions.append({"op": "must", "field": "source", "conds": [source]})
        if conditions:
            filters = {"op": "and", "conds": conditions}
            dsl_query["filter"] = filters
        result = []
        success = True
        for vec in vectors:
            ok, res, logid = self.client.recall(vec, topk=topk, index=index, dsl_query=dsl_query, **kwargs)
            if ok:
                tmp = []
                for r in res:
                    t = {}
                    t["label_lower64"] = r["label_lower64"]
                    t["label_upper64"] = r["label_upper64"]
                    t["scores"] = r["scores"]
                    t.update(json.loads(r["extra_infos"]))
                    tmp.append(t)
                res = tmp
            else:
                success = False
            result.append(res)
        return success, result


if __name__ == "__main__":
    t0 = time.time()
    config = {"vikingdb_name":"user_growth_1748420563__drama", "token":"ff0b99583c1716d24b358355a9800fdf", "region":"CN"}
    model = torch.load("/mnt/bn/vector2/model.pth")
    manager = VikingDBManager(**config)

    bid = "7433359455607082046"
    features, timestamp = manager.extract_video(vid="v0d8efg10000d0gdpufog65nis867sfg", shot_segment=True)
    res = manager.recall_vector(features, book_id=bid)
    for r, t in zip(res, timestamp):
        print(t, r[1][0]["episode_id"], r[1][0]["timestamp"], r[1][0]["scores"])

    # bid = "7433359455607082046"
    # bid_info = get_bid_info(bid)
    # for epid, vid in bid_info["ep_to_vid"].items():
    #     success, res = manager.extract_and_upload(
    #         vid=vid, 
    #         book_id=bid, 
    #         epid=epid, 
    #         source="fanqie", 
    #         shot_segment=True,
    #     )
    #     if success:
    #         print(f"第{epid}集上传成功")
    #     else:
    #         print(f"第{epid}集上传失败, 错误信息为{res}")

    # vector = torch.randn(512).tolist()
    # success, res = manager.recall_vector(vector, topk=3)
    # print(success, res)
    # msg, res = client.get_data({"label_upper64": 7433359455607082046, "label_lower64": 100131})

    # 第一集 v0debeg10000cskonb7og65o88gh7sgg
    # creative vid v0d8efg10000d0gdpufog65nis867sfg
    # script [{'epid': 2, 'start_time': 43.0/0, 'end_time': 146.452993/100}, {'epid': 3, 'start_time': 0/100, 'end_time': 125.0/224}]
