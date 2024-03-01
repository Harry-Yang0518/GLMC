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
STEP_SIZE=${3:-50}  # 默认为 50
GAMMA=${4:-0.5}  # 默认为 0.5

# 其他原有参数
MIXUP=${5:-'-1'}  # 你的mixup参数值
ALPHA=${6:-'1'}  # 你的mixup_alpha参数值


ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/hy2611/GLMC-LYGeng/
python main_wb.py --dataset cifar100 -a resnet32 --imbanlance_rate 1 --beta 0.5 --lr 0.01 \
--epochs 200 --loss ce --resample_weighting 0 --mixup ${MIXUP} --mixup_alpha ${ALPHA} \
--optimizer_type ${OPTIMIZER_TYPE} --scheduler_type ${SCHEDULER_TYPE} \
--step_size ${STEP_SIZE} --gamma ${GAMMA} \
--store_name ${OPTIMIZER_TYPE}_${SCHEDULER_TYPE}_ce_mx${MIXUP}_a${ALPHA}
"

#sbatch train_cifar100.sh SGD CosineAnnealingLR 50 0.5 -1 1
#sbatch train_cifar100.sh SGD StepLR 50 0.5 -1 1
#sbatch train_cifar100.sh SGD ExponentialLR 50 0.99 -1 1
#sbatch train_cifar100.sh Adam CosineAnnealingLR 50 0.5 -1 1
#sbatch train_cifar100.sh Adam StepLR 50 0.5 -1 1
#sbatch train_cifar100.sh Adam ExponentialLR 50 0.99 -1 1
#sbatch train_cifar100.sh RMSprop CosineAnnealingLR 50 0.5 -1 1
#sbatch train_cifar100.sh RMSprop StepLR 50 0.5 -1 1
#sbatch train_cifar100.sh RMSprop ExponentialLR 50 0.99 -1 1
