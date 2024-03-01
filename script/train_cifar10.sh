#!/bin/bash

#SBATCH --job-name=LTT_NC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

OPTIMIZER_TYPE=${1:-'SGD'}  # 默认为 'SGD'
SCHEDULER_TYPE=${2:-'CosineAnnealingLR'}  # 默认为 'CosineAnnealingLR'
STEP_SIZE=${3:-30}  # 默认为 30
GAMMA=${4:-0.1}  # 默认为 0.1

# 其他原有参数
MIXUP=${5:-'...'}  # 你的mixup参数值
ALPHA=${6:-'...'}  # 你的mixup_alpha参数值

singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/hy2611/GLMC-LYGeng/
python main_wb.py --dataset cifar10 -a resnet32 --imbanlance_rate 1 --beta 0.5 --lr 0.01 \
--epochs 200 --loss ce --resample_weighting 0 --mixup ${MIXUP} --mixup_alpha ${ALPHA} \
--optimizer_type ${OPTIMIZER_TYPE} --scheduler_type ${SCHEDULER_TYPE} \
--step_size ${STEP_SIZE} --gamma ${GAMMA} \
--store_name ${OPTIMIZER_TYPE}_${SCHEDULER_TYPE}_ce_mx${MIXUP}_a${ALPHA}
"