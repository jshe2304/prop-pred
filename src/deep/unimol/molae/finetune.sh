#!/bin/bash

data_path=$1
save_dir=$2
logfile=$save_dir/train.log

echo $logfile

n_gpu=4
MASTER_PORT=10086
dict_name="dict.txt"
weight_path="checkpoint_7_1000000.pt"
task_name="freesolv"  # molecular property prediction task name 
task_num=7
loss_func="finetune_smooth_mae"
lr=1e-4
batch_size=32
epoch=40
dropout=0
warmup=0.06
local_batch_size=32
only_polar=0
conf_size=11
seed=0

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

rm -rf ${save_dir}
mkdir -p ${save_dir}
mkdir -p ${save_dir}/tmp

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`

python -m torch.distributed.launch \
--nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
--task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
--conf-size $conf_size \
--num-workers 8 --ddp-backend=c10d \
--dict-name $dict_name \
--task mol_finetune --loss $loss_func --arch unimol_base  \
--classification-head-name $task_name --num-classes $task_num \
--optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
--lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
--update-freq $update_freq --seed $seed \
--fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
--log-interval 100 --log-format simple \
--validate-interval 1 \
--finetune-from-model $weight_path \
--best-checkpoint-metric $metric --patience 20 \
--save-dir $save_dir --only-polar $only_polar > ${logfile} 2>&1