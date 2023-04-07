#!/bin/bash
#SBATCH -J Polyp
#SBATCH -o ../SimT_step2.out
#SBATCH -e ../error.err
#SBATCH --gres=gpu:1
#SBATCH -w node30
#SBATCH --partition=team1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch020

cd /home/xiaoqiguo2/SimT_Plus/tools/
python -u ./trainV2_osn.py --lambda-WE 0.1 --lambda-Convex 0.1 --lambda-Volume 0.01 --lambda-Anchor 0.1 --num-classes 16 --open-classes 3 --learning-rate 1e-4 --learning-rate-T 1e-3 \
        --restore-from '../snapshots/SFDA/Synthia_warmup_iter118000_mIoU53.02.pth'