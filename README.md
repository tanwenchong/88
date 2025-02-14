# ENsiRNA
## Introduction
The source code of ENsiRNA.
## Installation
1.Clone ENsiRNA to a local directory 
```
git clone git@github.com:tanwenchong/88.git
```
2.Download Rosetta from (https://www.rosettacommons.org/software/license-and-download) ,then change the Rosetta dir in ../88/ENsiRNA/data/get_pdb.py and ../88/ENsiRNA-mod/data/get_pdb.py      

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
## Usage
For ENsiRNA
```
cd ../88/ENsiRNA
```
For ENsiRNA-mod
```
cd ../88/ENsiRNA-mod
```
## Data process
The format of your_file.csv and your_file.xlsx can be referenced from the provided dataset.
- For ENsiRNA
```
python -m data.get_pdb -f your_file.csv -p pdb_path #ENsiRNA
```
- For ENsiRNA-mod
```
python -m data.get_pdb -f your_file.excel -p pdb_path #ENsiRNA-mod
```
The pre-folded PDB files for ENsiRNA are available at https://drive.google.com/file/d/1XHuFuqW7s93lBmCrZH70jN-41hmsF071/view?usp=drive_link
The pre-folded PDB files for ENsiRNA-mod are available at https://drive.google.com/file/d/1F7cNJXMNPSjFb0UvDkDRHTkt9Tt4EGWe/view?usp=drive_link
## training
You need to change your dataset dir in config.json according to the output from Data process  
For both ENsiRNA and ENsiRNA-mod:
```
GPU=0 bash train.sh config.json
```
## test
your_file.json is from Data process
For both ENsiRNA and ENsiRNA-mod
```
GPU=0 bash test.sh your_file.json saving_path
```
## Pipeline
We provide a tool for predicting siRNAs based on mRNA sequences, which is easy to use and test.
```
cd ../88/ENsiRNA
bash design_pipeline.sh Your_mRNA.fasta Work_dir
```
The results will be saved as `result.xlsx` in your specified `Work_dir`. You can use the mean scores as a reference for design decisions.
The file contains:
- Predicted siRNA sequences
- 5-fold cross-validation models result

