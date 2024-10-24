# ENsiRNA
## Introduction
The source code of ENsiRNA.
## Installation
1.Clone ENsiRNA to a local directory    
2.Download Rosetta from (https://www.rosettacommons.org/software/license-and-download) ,then change the Rosetta dir in data.get_pdb     

3.Create conda environment and prepare the required software
```
conda create -n my_environment_name python=3.10
conda activate my_environment_name
conda install viennarna=2.6.4-0
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
5.Change the path in get_pdb.py
## Data process
```
python -m data.get_pdb -f your_file.csv -p pdb_path #ENsiRNA
python -m data.get_pdb -f your_file.excel -p pdb_path #ENsiRNA-mod
```
The folding PDB files for ENsiRNA are available at https://drive.google.com/file/d/1XHuFuqW7s93lBmCrZH70jN-41hmsF071/view?usp=drive_link
The folding PDB files for ENsiRNA-mod are available at https://drive.google.com/file/d/1F7cNJXMNPSjFb0UvDkDRHTkt9Tt4EGWe/view?usp=drive_link
## training
```
GPU=0 bash train.sh config.json
```
## test
```
GPU=0 bash test.sh your_file.json saving_path
```
