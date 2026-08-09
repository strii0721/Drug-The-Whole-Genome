
# ============================================
# 双机多机版（blue-whale + sperm-whale）
# 两只鲸鱼各 1 张 NVIDIA GB10（aarch64），经 ConnectX-7 直连：
#   blue-whale   = 10.100.0.1
#   sperm-whale  = 10.100.0.3
#   NCCL 专用接口: enp1s0f0np0
# 两机各跑同一条命令；数据/代码在两机各备一份（scripts/sync_to_whales.sh）。
# 两机直接跑同一条命令，torchrun 自动分配 rank；先启动的节点成为 rank0（建议 blue-whale 先启动）。
# ============================================
data_path="./resources/datasets/PDBbind_train"

save_dir="./output/pretrain/$(date +"%Y-%m-%d_%H-%M-%S")/"
tmp_save_dir="./output/pretrain/tmp/$(date +"%Y-%m-%d_%H-%M-%S")/"
tsb_dir="./output/pretrain/tsb_dir/$(date +"%Y-%m-%d_%H-%M-%S")/"

# 每机 1 卡（GB10 单卡）
n_gpu=1
# 多机参数：默认 2 节点，RANK0 为 blue-whale（10.100.0.1）
# 单机冒烟：NNODES=1 bash scripts/pretrain.sh
NNODES=${NNODES:-2}
RANK0_IP=10.100.0.1
MASTER_PORT=10055
# NCCL 走 ConnectX-7 专用接口（enp1s0f0np0），防止误走其它网卡
export NCCL_SOCKET_IFNAME=enp1s0f0np0
finetune_mol_model="./resources/model_weights/unimol/mol_pre_no_h_220816.pt"
finetune_pocket_model="./resources/model_weights/unimol/pocket_pre_220816.pt"

batch_size=18
batch_size_valid=18
epoch=200
dropout=0.0
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3
lr=1e-3

mkdir -p logs/pretrain
exec > >(tee "logs/pretrain/$(date +%s).log") 2>&1

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
# 启动模式切换：NNODES>=2 为双机多机（两机各跑同一条命令，torchrun 自动分配 rank，
# RANK0/blue-whale 自动拉起 c10d rendezvous）；NNODES=1 为单机冒烟
# （master_addr 默认 127.0.0.1）。切换方式：NNODES=1 bash scripts/pretrain.sh
if [[ "$NNODES" -ge 2 ]]; then
    LAUNCH_ARGS="--nnodes=$NNODES --nproc_per_node=$n_gpu --rdzv_endpoint=$RANK0_IP:$MASTER_PORT --rdzv_backend=c10d"
else
    LAUNCH_ARGS="--nproc_per_node=$n_gpu --master_port=$MASTER_PORT"
fi

CUDA_VISIBLE_DEVICES="0" torchrun $LAUNCH_ARGS $(which unicore-train) $data_path --user-dir ./src/unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 20 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-last-epochs 5 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model