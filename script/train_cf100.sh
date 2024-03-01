#!/bin/bash

#SBATCH --job-name=lt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
LOSS=${1}
CENTER=${2}
L2NORM=${3}
FNORM=${4}
# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/hy2611/GLMC-LYGeng/
python main_wb.py --dataset cifar100 -a resnet32 --imbanlance_rate 1 --beta 0.5 --lr 0.01 \
--epochs 200 --loss ${LOSS} --center ${CENTER} --l2norm ${L2NORM} --aug none \
--resample_weighting 0 --mixup -1 --mixup_alpha 1 --fnorm ${FNORM} \
--store_name ${LOSS}_center_${CENTER}_l2_${L2NORM}_fn_${FNORM}
 " 