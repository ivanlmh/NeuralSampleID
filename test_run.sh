#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12          # request cores per GPU
#$ -l h_rt=36:0:0    # hours runtime
#$ -l h_vmem=11G      #  RAM per core
#$ -l gpu=1           # request GPU
#$ -l gpu_type=ampere
#$ -N ivan_test_grafp_tc29_m
#$ -o /data/home/eez083/NeuralSampleID/output/ivan_test
#$ -m beas

# module load python/3.10.7
# source ../grafp_venv/bin/activate
source /data/home/eez083/.bashrc
conda activate neural_sample_id
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=10 --test_dir=/data/EECS-Studiosync/fma_small --text=ivan_test_fma_small
python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=10 --test_dir=/data/home/eez083/sample_100 --text=ivan_test_sample_100