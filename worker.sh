export TCE_CLUSTER=tcr
export TCE_PSM=toutiao.growth.dvr
export PYTHONPATH=/opt/tiger/drama_training

python3 atom/modules/vector_retrieval/modules/worker.py --concurrent 1


# chmod +x /opt/tiger/drama_training/atom/modules/vector_retrieval/worker.sh
# sudo seq 4 | xargs -I {} -P 4 /opt/tiger/drama_training/atom/modules/vector_retrieval/worker.sh