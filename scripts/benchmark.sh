TASK="DUDE" # DUDE or PCBA

results_path="output/test_$TASK"  # auto-generated from TASK name
batch_size=64

# 临时用官方权重跑通验证；新训练权重到位后改回 drug_clip_re/checkpoint_best.pt
# weight_path="./resources/model_weights/drug_clip_re/checkpoint_best.pt"
weight_path="./resources/model_weights/benchmark/dude_ecfp_90.pt"
use_folds=False

log_dir="output/benchmark"
mkdir -p $log_dir
log_file="$log_dir/${TASK}-$(date +%Y-%m-%d_%H-%M-%S).log"

CUDA_VISIBLE_DEVICES="0" python ./src/unimol/test.py --user-dir ./src/unimol "./resources/dict" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --use-folds $use_folds \
       --path $weight_path \
       --benchmark-data-dir ./data \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK 2>&1 | tee $log_file