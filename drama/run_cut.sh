export PYTHONPATH=/opt/tiger/poron/:/opt/tiger/argon/:/opt/tiger/:$PYTHONPATH
export TCE_PSM=toutiao.growth.zinc
export TCE_CLUSTER=default
cd /opt/tiger/atom
python -m servers.drama.cut_video
