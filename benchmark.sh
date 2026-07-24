TASK="PCBA" # DUDE or PCBA

results_path="output/test_$TASK"  # auto-generated from TASK name
batch_size=12

weight_path="./output/pretrain/2026-07-24_03-16-10/checkpoint_best.pt"
use_folds=False

CUDA_VISIBLE_DEVICES="0" python ./unimol/test.py --user-dir ./unimol "./dict" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --use-folds $use_folds \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \