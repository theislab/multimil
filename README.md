# Multigrate: multi-omic data integration and transformation for signle-cell genomics 

To start:
1. Set up a Conda environment:
`conda create -n scmulti python=3.7`
and `conda activate scmulti`

2. Install the correct version of scanpy to work with scIB
`pip install  scanpy==1.4.6`

3. Install scIB
`pip install git+https://github.com/theislab/scib.git`

4. Install Jupyter notebooks
`conda install -c conda-forge notebook`

5. Install pytorch
`conda install pytorch torchvision -c pytorch`

6. Install a different version of louvain
`pip uninstall louvain` and
`conda install conda-forge louvain`

7. Install scmulti
`pip install git+https://github.com/theislab/multigrate.git@train-method`

Additionally:
1. Download the dataset of interest from [Drive](https://drive.google.com/drive/u/0/folders/1vdO8CJluRp7sOOQHc85YYZXhV2p2QcDN)

2. Run one of the notebooks from the example folder
