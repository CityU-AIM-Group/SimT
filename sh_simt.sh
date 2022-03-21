#!/bin/bash
#SBATCH -J Polyp
#SBATCH -o SimT_BAPA1.out    
#SBATCH -e error.err
#SBATCH --gres=gpu:1
#SBATCH -w node2

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch020

cd /home/xiaoqiguo2/SimT/tools/
python -u ./trainV2_simt.py --open-classes 15 --learning-rate 6e-4 --learning-rate-T 6e-3 --Threshold-high 0.8 --Threshold-low 0.2 --lambda-Place 0.1 --lambda-Convex 0.1 --lambda-Volume 1.0 --lambda-Anchor 1.0 --restore-from '../snapshots/GTA5_BAPA_warmup_iter129000_mIoU57.44.pth'