#!/bin/bash
#SBATCH -J City
#SBATCH -o ../SimT_step3.out
#SBATCH -e ../error.err
#SBATCH --gres=gpu:1
#SBATCH -w node3

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch020

cd /home/xiaoqiguo2/SimT_Plus/tools/
python -u ./trainV3_ntr.py --temperature 6. --lambda-clean 0.5 --lambda-relation 0.1 --num-classes 16 --open-classes 3 --learning-rate 1e-4 \
        --restore-from '../snapshots/SimT/Synthia_iter1500_mIoU57.51.pth' \
        --restore-from-noise '../snapshots/GTA5_BAPA_warmup_iter124000_mIoU55.11.pth'