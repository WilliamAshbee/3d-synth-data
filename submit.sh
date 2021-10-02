#!/bin/bash
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=32g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:v100:1
#SBATCH -t 600
#SBATCH -J wa-mutico
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 10s

export PATH=/data/users2/washbee/anaconda3/bin:$PATH
source /data/users2/washbee/anaconda3/etc/profile.d/conda.sh
conda activate /data/users2/washbee/anaconda3/envs/c3d-synthd
python 3d_icomeba_predict.py 

sleep 30s
