#!/bin/bash
#SBATCH --job-name=hf_zero_shot
#SBATCH --output=/home/asoubki/scratch/logs/%x.%j.out
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=

PYTHON=/home/asoubki/.miniconda3/envs/hedging/bin/python
BIN=/home/asoubki/dev/hedging/bin/hf_zero_shot.py
$PYTHON $BIN $CONFIG
