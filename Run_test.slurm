#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition shortgpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 00:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64000

module load Anaconda3
source activate myenvpy27Clone

python /data/cephfs/punim0676/DLinCV-master/scripts/keras-posenet-Debaditya/LSTMBIMPoseNetWorkingOtherDatasets/edgeRender/train.py