# EnMod
## Introduction
We appreciate the code and inspiration from dyMEAN.
## Installation
1.Clone EnmodSiRNA to a local directory    
2.Download Rosetta from (https://www.rosettacommons.org/software/license-and-download) ,then change the Rosetta dir in data.get_pdb     

3.Create conda environment and prepare the required software
```
conda create -n my_environment_name python=3.10
conda activate my_environment_name
conda install viennarna
```
4.Pip install packages  
```
pip install biopython numpy pandas scipy tensorboard tqdm openpyxl rdkit scikit-learn xgboost
# install pytorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
#install pretrained RNA LLM
pip install rna-fm
```
## Data process
```
python -m data.get_pdb -f your_file.excel -p pdb_path
```
## training
```
GPU=0 bash train.sh config.json
```
## test
```
GPU=0 bash test.sh your_file.json
```
