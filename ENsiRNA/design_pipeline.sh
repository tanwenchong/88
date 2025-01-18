#!/bin/bash
#bash design_pipeline.sh /public2022/tanwenchong/rna/EnSIRNA-main/dulab/AURKB/AURKB.fasta /public2022/tanwenchong/rna/EnSIRNA-main/dulab/AURKB 1 1
mrna_fasta=$1
mrna_fasta_name=$(basename "$mrna_fasta" | cut -f1 -d".")
work_path=$2

#是否获取数据
get_data=${3:-1}
#是否运行模型
run_model=${4:-1}  # 如果没有第三个参数，默认为0


if [ $get_data == 1 ]; then
    #接收get_siRNA.py的输出
    csv_path=$(python get_siRNA.py -i $mrna_fasta -o $work_path)
    #创建pdb文件夹 $mrna_fasta_name+_pdb
    pdb_path=$work_path/$mrna_fasta_name"_pdb"
    #如果pdb_path不存在，则创建
    if [ ! -d $pdb_path ]; then
        mkdir -p $pdb_path
    fi
    python -m data.get_pdb -f $csv_path -p $pdb_path


    
fi
json_path=$work_path/$mrna_fasta_name.json
CKPT1='/public2022/tanwenchong/rna/EnSIRNA-main/919/model_pkl/version_0/checkpoint/epoch36_step5217.ckpt'
CKPT2='/public2022/tanwenchong/rna/EnSIRNA-main/919/model_pkl/version_1/checkpoint/epoch33_step4794.ckpt'
CKPT3='/public2022/tanwenchong/rna/EnSIRNA-main/919/model_pkl/version_2/checkpoint/epoch28_step4089.ckpt'
CKPT4='/public2022/tanwenchong/rna/EnSIRNA-main/919/model_pkl/version_3/checkpoint/epoch33_step4794.ckpt'
CKPT5='/public2022/tanwenchong/rna/EnSIRNA-main/919/model_pkl/version_4/checkpoint/epoch19_step2820.ckpt'

#export CUDA_VISIBLE_DEVICES=2
if [ $run_model == 1 ]; then    
    python run.py \
        --ckpt ${CKPT1} ${CKPT2} ${CKPT3} ${CKPT4} ${CKPT5}\
        --test_set $json_path\
        --save_dir $work_path\
        --gpu 0
fi
