# ENsiRNA
## Introduction
The source code of ENsiRNA.
![](Figure1.png)
## Installation
Two methods exist to run ENsiRNA:  
1. Docker (Recommended)  
2. Linux  
### 1. Docker (Recommended) 
We provide a Docker image that enables cross-platform execution of ENsiRNA without the need to install dependencies.  
Note: The Docker requires support for NVIDIA GPU with CUDA 11.8 and a network connection for downloading weights for RNA-FM.  

#### Download Docker image from DockerHub OR aliyun:   
```bash
#From DockerHub
docker pull tanwenchong/ensirna:v2

#From aliyun 
docker pull crpi-tv4nd4fiip8xechs.cn-guangzhou.personal.cr.aliyuncs.com/ensirna/ensirna:v2
```
#### Create a container：
##### On Linux (bash):
```bash
#From DockerHub
cmd=$(docker run --gpus all -it -d tanwenchong/ensirna:v2)

#From aliyun
cmd=$(docker run --gpus all -it -d crpi-tv4nd4fiip8xechs.cn-guangzhou.personal.cr.aliyuncs.com/ensirna/ensirna:v2)
```
##### On Windows PowerShell:
```
#From DockerHub
$cmd = docker run --gpus all -it -d tanwenchong/ensirna:v2

#From aliyun
$cmd = docker run --gpus all -it -d crpi-tv4nd4fiip8xechs.cn-guangzhou.personal.cr.aliyuncs.com/ensirna/ensirna:v2
```
### 2. Linux  
#### 1. Clone the repository
First, clone the ENsiRNA repository to your local machine:
```bash
git clone git@github.com:tanwenchong/ENsiRNA.git
```
#### 2. Download Rosetta
Download Rosetta from the official website: (https://www.rosettacommons.org/software/license-and-download).
Once downloaded, change the Rosetta directory in the following files:
- `./ENsiRNA-main/ENsiRNA/data/get_pdb.py`
- `./ENsiRNA-main/ENsiRNA-mod/data/get_pdb.py`    
#### 3. Create a Conda environment
Create a new Conda environment and install the required software:
```bash
conda create -n my_environment_name python=3.10
conda activate my_environment_name
conda install viennarna=2.6.4-0
```
#### 4. Install Python dependencies
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

### ENsiRNA
The usage here is mainly for Docker, the detailed training and testing on linux is in [sub-folder](https://github.com/tanwenchong/ENsiRNA/tree/main/ENsiRNA).  

#### Copy prepared input local mRNA fasta file <mrna.fasta> to the container:  
##### On Linux (bash):
```bash
#replace <mrna.fasta> with your local file path
docker cp <mrna.fasta>  $cmd:/app/ENsiRNA-main/ENsiRNA/mrna.fasta
```
##### On Windows PowerShell:
```bash
docker cp <mrna.fasta>  ${cmd}:/app/ENsiRNA-main/ENsiRNA/mrna.fasta
```

#### Enter the container:  
```bash
docker exec -it $cmd bash
```
#### Run the program:
```bash
conda activate my_environment_name
cd /app/ENsiRNA-main/ENsiRNA
bash design.sh /app/ENsiRNA-main/ENsiRNA/mrna.fasta /app/ENsiRNA-main/ENsiRNA/result
```
```
Usage: bash design.sh [options] <mrna_fasta> <work_path>

Arguments:
  <mrna_fasta>    Path to the mRNA FASTA file
  <work_path>     Directory to save all output files
```

The result is in /app/ENsiRNA-main/ENsiRNA/result/mrna_result.xlsx  
When you complete your analysis, copy any desired output files off the container to your local machine with the docker cp command. Shut down and clean up your container like this:  
(Ctrl + P + Q) to exit
```
docker cp $cmd:/app/ENsiRNA-main/ENsiRNA/result/mrna_result.xlsx <local file dir> #For Linux
docker cp ${cmd}:/app/ENsiRNA-main/ENsiRNA/result/mrna_result.xlsx <local file dir> #For Windows PowerShell
docker stop $cmd
docker rm $cmd
```
### ENsiRNA-mod
The usage here is mainly for Docker, the detailed training and testing on linux is in [sub-folder](https://github.com/tanwenchong/ENsiRNA/tree/main/ENsiRNA-mod).  
```bash
conda activate my_environment_name
cd /app/ENsiRNA-main/ENsiRNA-mod
python easy_run.py
```
With directly running easy_run.py, you can easy to predict flowing the instruction in pipeline. There are examples in the pipeline, the full modification type can be found at [ENsiRNA-mod/data/mod_utils.py](https://github.com/tanwenchong/ENsiRNA/blob/main/ENsiRNA-mod/data/mod_utils.py)
```
usage: python easy_run.py [-h] [-o OUTPUT_DIR] [--model MODEL]

ENsiRNA: A pipeline for modified siRNA prediction

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to save results (default: 'result')
  --model MODEL         Path to model checkpoint (default: pkl/checkpoint_1.ckpt)
```

### PDB avaliable
The pre-folded PDB files for ENsiRNA are available at https://drive.google.com/file/d/1XHuFuqW7s93lBmCrZH70jN-41hmsF071/view?usp=drive_link
The pre-folded PDB files for ENsiRNA-mod are available at https://drive.google.com/file/d/1F7cNJXMNPSjFb0UvDkDRHTkt9Tt4EGWe/view?usp=drive_link

### License
ENsiRNA is released under an Apache v2.0 license.





