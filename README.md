# ENsiRNA
## Introduction
The source code of ENsiRNA.
## Installation
### 1. Clone the repository
First, clone the ENsiRNA repository to your local machine:
```bash
git clone git@github.com:tanwenchong/88.git
```

### 2. Download Rosetta
Download Rosetta from the official website: (https://www.rosettacommons.org/software/license-and-download).
Once downloaded, change the Rosetta directory in the following files:
- `../88/ENsiRNA/data/get_pdb.py`
- `../88/ENsiRNA-mod/data/get_pdb.py`    

### 3. Create a Conda environment
Create a new Conda environment and install the required software:
```bash
conda create -n my_environment_name python=3.10
conda activate my_environment_name
conda install viennarna=2.6.4-0
```

### 4. Install Python dependencies
Use pip to install the necessary Python packages:
```bash
pip install biopython numpy pandas scipy tensorboard tqdm openpyxl rdkit scikit-learn xgboost
```

For PyTorch with CUDA 11.8 support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

Install the pretrained RNA LLM model:
```bash
pip install rna-fm
```

## Usage
### For ENsiRNA
```bash
cd ../88/ENsiRNA
```

### For ENsiRNA-mod
```bash
cd ../88/ENsiRNA-mod
```

## Data Processing

The format of `your_file.csv` and `your_file.xlsx` should follow the structure of the provided dataset. 

- **For ENsiRNA**:
  (Users should format their data into `your_file.xlsx` as per the example.)
  ```bash
  python -m data.get_pdb -f your_file.csv -p pdb_path # ENsiRNA
  ```

- **For ENsiRNA-mod**:
  (For modification types, users can search or add data in `data/mod_utils.py`. Format your data into `your_file.xlsx` as per the example.)
  ```bash
  python -m data.get_pdb -f your_file.xlsx -p pdb_path # ENsiRNA-mod
  ```
### Pre-folded PDB Files
The pre-folded PDB files for ENsiRNA are available at https://drive.google.com/file/d/1XHuFuqW7s93lBmCrZH70jN-41hmsF071/view?usp=drive_link
The pre-folded PDB files for ENsiRNA-mod are available at https://drive.google.com/file/d/1F7cNJXMNPSjFb0UvDkDRHTkt9Tt4EGWe/view?usp=drive_link

## Training

Before training, ensure you’ve updated the dataset directory in `config.json` according to the output from the data processing step.

For both **ENsiRNA** and **ENsiRNA-mod**, use the following command:
```bash
GPU=0 bash train.sh config.json
```

## Testing

Test the model using the output file (`your_file.json`) from the data processing step.

For both **ENsiRNA** and **ENsiRNA-mod**:
```bash
GPU=0 bash test.sh your_file.json saving_path
```

## Pipeline

The tool also provides a pipeline for predicting siRNAs based on mRNA sequences.

To run the pipeline, use the following command:
```bash
cd ../88/ENsiRNA
bash design_pipeline.sh Your_mRNA.fasta Your_dir
```

### Output
The results will be saved as `result.xlsx` in the specified directory (`Your_dir`). The file will contain:
- Predicted siRNA sequences
- 5-fold cross-validation model results
```

