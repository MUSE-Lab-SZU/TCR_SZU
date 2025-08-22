from atom.modules.vector_retrieval.modules.utils import (
    upload_book,
    query_book,
    upload_video,
    recall_video,
    post_process,
    filter_candidates,
)

if __name__ == "__main__":
    import json

    # vid="v0debeg10000cskonb7og65o88gh7sgg"
    # vid="v0d8efg10000d0gdpufog65nis867sfg"
    vid="v0d8efg10000d0dp2q7og65u7hg0se1g"
    # vid="v028efg10000d0dp4jvog65ineenn72g"
    # vid="v028efg10000d0ee2nnog65l7lsdpv9g"
    # vid="v0d8efg10000d0f3f67og65j8ud1llp0"
    # vid="v038efg10000d0e3t4nog65ts1o461pg"
    # vid = "v0d8efg10000cn1lqj3c77u8sth8ikb0"
    # vid = "v028efg10000cn8q3kbc77u8q6scpm0g"
    # vid = "v028efg10000cn3h1vjc77u46hvomdr0"
    # vid = "v0debeg10000cskon97og65h132ulang"

    bid = "7433359455607082046"
    topk = 3
    use_cache = True
    # epid = 1

    # task_list = upload_book(bid, max_interval=2, use_cache=use_cache)
    # query_book(bid, task_list)

    # print(upload_video(vid=vid, bid=bid, epid=epid, use_cache=use_cache))

    res = recall_video(vid=vid, bid=bid, topk=topk, max_interval=2, use_cache=use_cache)
    value = json.loads(res.extra_info)
    raw = value["result"]
    print("原始召回")
    for r in raw:
        print(r)
        print()
    candidates = filter_candidates(raw, beam_size=topk)
    print("过滤候选镜头")
    for r in candidates:
        print(r)
    first = post_process(candidates, 3500)
    print("一阶聚合")
    for r in first:
        print(r)
    second = post_process(first, 3500)
    print("二阶聚合")
    for r in second:
        print(r)
