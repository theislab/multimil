#!/bin/bash
#PBS -N scmulti-validate
#PBS -m abe
#PBS -M alireza530@gmail.com
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q tensorflow

# setup conda
source ~/.bashrc
conda activate scanpy

# run the script
cd $PBS_O_WORKDIR
python -u scmulti/validate.py --root-dir "outputs/$exp"
