export TCE_PSM=toutiao.growth.zinc
export TCE_CLUSTER=default
doas -p toutiao.growth.zinc python atom/modules/vector_retrieval/data/build_data.py
