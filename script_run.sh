#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12          # 16 cores per GPU
#$ -l h_rt=40:0:0    # 40 hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -l gpu=1           # request 2 GPU
#$ -l gpu_type=ampere
#$ -N ivan_1st_train_grafp
#$ -o /data/home/eez083/NeuralSampleID/output/ivan_script
#$ -m beas

# module load python/3.10.7
# source ../grafp_venv/bin/activate
# python train.py --ckp=tc_30

source /data/home/eez083/.bashrc
conda activate neural_sample_id

python separate_audio.py