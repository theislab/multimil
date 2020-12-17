#!/bin/bash
#SBATCH --job-name=scmulti-trainval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --gpus=1

# validate arguments
config_file=$1
output_base_dir=$2
config_filename=$(basename -- "$1")
experiment_name="${config_filename%.*}"
output_dir=$2/$experiment_name

# setup conda
source "/opt/miniconda3/etc/profile.d/conda.sh"
conda activate scanpy

# run the script
python -u scmulti/train.py --config-file $config_file --output-dir $output_dir && \
python -u scmulti/validate.py --root-dir $output_dir