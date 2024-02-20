#!/bin/bash
#SBATCH --job-name=yolo8_m_SGD_b25_mix0_raw_sigClas
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=SPG-1-[1-4]

source ~/.bashrc
conda activate yolo

export CUDA_VISIBLE_DEVICES=0

yolo detect train data=./seacu_yolo8_hpc.yaml \
    device=0 model=yolov8m.pt workers=1 epochs=500 batch=25 optimizer='SGD' \
    lr0=0.001 single_cls=True patience=0

