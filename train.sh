export WORK_DIR="./job_data/work_dirs/"
export GPUS=8
export PYTHONPATH="${PYTHONPATH}:~/waytous/cjj/waytous"
export CONFIG="./projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
# CUDA_VISIBLE_DEVICES=6,7 
python -m torch.distributed.run \
    --nproc_per_node=${GPUS} \
    --master_port=2333 \
    tools/train.py ${CONFIG} \
    --launcher pytorch \
    --deterministic \
    --work-dir ${WORK_DIR}

# 明天让occ加载训练好的权重再跑一遍