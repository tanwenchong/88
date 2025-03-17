#!/bin/bash

mrna_fasta=$1
mrna_fasta_name=$(basename "$mrna_fasta" | cut -f1 -d".")
work_path=$2


get_data=${3:-1}

run_model=${4:-1}  


if [ $get_data == 1 ]; then

    csv_path=$(python get_siRNA.py -i $mrna_fasta -o $work_path)

    pdb_path=$work_path/$mrna_fasta_name"_pdb"

    if [ ! -d $pdb_path ]; then
        mkdir -p $pdb_path
    fi
    python -m data.get_pdb -f $csv_path -p $pdb_path

fi
echo "pdb files generated"
json_path=$work_path/$mrna_fasta_name.json
CKPT1='pkl/checkpoint_1.ckpt'
CKPT2='pkl/checkpoint_2.ckpt'
CKPT3='pkl/checkpoint_3.ckpt'
CKPT4='pkl/checkpoint_4.ckpt'
CKPT5='pkl/checkpoint_5.ckpt'

echo "start running model"
if [ $run_model == 1 ]; then    
    python run.py \
        --ckpt ${CKPT1} ${CKPT2} ${CKPT3} ${CKPT4} ${CKPT5}\
        --test_set $json_path\
        --save_dir $work_path\
        --gpu 0
        --id $mrna_fasta_name
fi

echo "ENsiRNA design pipeline completed"
echo "result saved in $work_path"
