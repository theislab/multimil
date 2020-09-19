#!/bin/bash
#PBS -N scmulti-train-val
#PBS -m abe
#PBS -M alireza530@gmail.com
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q SP

# setup conda
source ~/.bashrc
conda activate scanpy

# run the script
cd $PBS_O_WORKDIR
python -u scmulti/train.py --config-file "experiments/$exp.json" && python -u scmulti/validate.py --root-dir "outputs/$exp"
