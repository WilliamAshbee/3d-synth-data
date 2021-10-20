#!/bin/bash
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=30g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:v100:1
#SBATCH -t 6000
#SBATCH -J wa-mutico
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

export PATH=/data/users2/washbee/anaconda3/bin:$PATH
source /data/users2/washbee/anaconda3/etc/profile.d/conda.sh
conda activate /data/users2/washbee/anaconda3/envs/c3d-synthd
python vit3d_train.py --arg1 $SLURM_ARRAY_TASK_ID --arg2 0 &
python vit3d_train.py --arg1 $SLURM_ARRAY_TASK_ID --arg2 1 &
gpustat -i 100 &
wait

sleep 10s
