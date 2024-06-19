#!/bin/bash

#SBATCH --job-name=bn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
# IB=$1
# LOSS=$2
# FEAT=$3


# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/hy2611/GLMC/
python main_bn.py --dataset cifar100 -a mresnet32 --imbalance_rate 0.01 \
--imbalance_type exp --lr 0.01 --seed 2021 --epochs 200 --loss ce \
--branch2 --contrast --bias --contrast_weight 4 \
--weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 \

--store_name cf100_exp_baseline "

# python main_bn.py --batch_size 64 --dataset cifar10 -a mresnet32 --imbalance_rate ${IB} --imbalance_type step --lr 0.01 --seed 2021 
# --epochs 200 --loss ${LOSS} --feat none --bn_type bn 
# --resample_weighting 0 --store_name batchn_IR_${IB}_${LOSS}_noetf "

# python main_bn.py --dataset cifar10 -a mresnet32 --imbalance_rate 0.01 --imbalance_type step --lr 0.01 --seed 2021 --epochs 200 --loss ce --bn_type bn

