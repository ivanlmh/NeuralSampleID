#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12          # request cores per GPU
#$ -l h_rt=36:0:0    # hours runtime
#$ -l h_vmem=6G      #  RAM per core 11?
#$ -l gpu=1           # request GPU
#$ -l gpu_type=ampere
#$ -N ivan_test_grafp_tc29_m
#$ -o /data/home/eez083/NeuralSampleID/output/ivan_test
#$ -m beas

# module load python/3.10.7
# source ../grafp_venv/bin/activate
source /data/home/eez083/.bashrc
conda activate neural_sample_id
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=-5 --test_dir=/data/EECS-Studiosync/fma_small --text=ivan_test_fma_small-5snr
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=-5 --test_dir=/data/EECS-Studiosync/fma_small --text=ivan_test_fma_small_sample100noise-5snr
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=10 --test_dir=/data/EECS-Studiosync/fma_small --text=ivan_test_fma_small_sample100noise
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=10 --test_dir=/data/home/eez083/sample_100 --text=ivan_test_sample_100
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=10 --test_dir=/data/ECS-Studiosync/fma_small --sample_id --text=ivan_test_fma_small_sample_id

# python test_fp.py --query_lens=1,2,3,5 --n_query_db=100 --n_dummy_db=800 --test_snr=10 --test_dir=/home/ivan/Documents/FIng/FRANCE_OLD/PRIM/DATASETS/fma_small --sample_dir=/home/ivan/Documents/FIng/QueenMary/sample_100 --sample_id --text=ivan_test_fma_small_sample_id
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=100 --n_dummy_db=800 --test_snr=10 --test_dir=/data/home/eez083/sample_100 --sample_dir=/data/home/eez083/sample_100 --sample_id --text=ivan_sample_id_first_attempt2
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=100 --n_dummy_db=800 --test_snr=-5 --test_dir=/data/EECS-Studiosync/fma_small --sample_dir=/data/home/eez083/sample_100 --sample_id --text=ivan_sample_id_first_attempt3
# python test_fp.py --query_lens=1,2,3,5 --n_query_db=100 --n_dummy_db=800 --test_snr=10 --test_dir=/data/home/eez083/sample_100 --sample_dir=/data/home/eez083/sample_100 --sample_id --text=ivan_sample_id_first_attempt4

python test_sampleID.py --query_lens=1,2,3,5 --n_query_db=100 --n_dummy_db=400 --test_snr=10 --test_dir=/data/EECS-Studiosync/fma_small --sample_dir=/data/home/eez083/sample_100 --text=ivan_sample_id_indipendent_target
