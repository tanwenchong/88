#!/bin/bash

# ENsiRNA Design Pipeline
# This script automates the process of designing effective siRNAs using the ENsiRNA model

# Display help information
show_help() {
    echo "Usage: $0 [options] <mrna_fasta> <work_path>"
    echo ""
    echo "Arguments:"
    echo "  <mrna_fasta>    Path to the mRNA FASTA file"
    echo "  <work_path>     Directory to save all output files"
    echo ""
    echo "Options:"
    echo "  -h, --help      Display this help message and exit"
    echo "  -s, --skip-data Skip data preparation step (default: run data preparation)"
    echo "  -m, --skip-model Skip model execution step (default: run model)"
    echo "  -g, --gpu GPU   Specify GPU device number (default: 0)"
    echo ""
    echo "Example:"
    echo "  $0 input.fasta ./results"
    echo "  $0 --skip-data input.fasta ./results"
    exit 1
}

# Initialize default parameters
get_data=1
run_model=1
gpu_id=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -s|--skip-data)
            get_data=0
            shift
            ;;
        -m|--skip-model)
            run_model=0
            shift
            ;;
        -g|--gpu)
            gpu_id=$2
            shift 2
            ;;
        *)
            # Store positional arguments
            if [ -z "$mrna_fasta" ]; then
                mrna_fasta=$1
            elif [ -z "$work_path" ]; then
                work_path=$1
            else
                echo "Error: Unexpected argument '$1'"
                show_help
            fi
            shift
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$mrna_fasta" ] || [ -z "$work_path" ]; then
    echo "Error: Missing required arguments"
    show_help
fi

# Check if input file exists
if [ ! -f "$mrna_fasta" ]; then
    echo "Error: Input file '$mrna_fasta' does not exist"
    exit 1
fi

# Extract filename without extension
mrna_fasta_name=$(basename "$mrna_fasta" | cut -f1 -d".")

# Create work directory if it doesn't exist
if [ ! -d "$work_path" ]; then
    echo "Creating output directory: $work_path"
    mkdir -p "$work_path"
fi

echo "=== ENsiRNA Design Pipeline ==="
echo "Input: $mrna_fasta"
echo "Output directory: $work_path"

# Data preparation step
if [ $get_data -eq 1 ]; then
    echo "[1/2] Preparing siRNA data..."
    
    csv_path=$(python get_siRNA.py -i $mrna_fasta -o $work_path)
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate siRNA data"
        exit 1
    fi

    pdb_path=$work_path/${mrna_fasta_name}_pdb
    if [ ! -d $pdb_path ]; then
        mkdir -p $pdb_path
    fi
    
    echo "Generating PDB files..."
    python -m data.get_pdb -f $csv_path -p $pdb_path
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate PDB files"
        exit 1
    fi
    
    echo "✓ Data preparation completed"
else
    echo "[1/2] Skipping data preparation step"
fi

# Model execution step
json_path=$work_path/$mrna_fasta_name.json
CKPT1='pkl/checkpoint_1.ckpt'
CKPT2='pkl/checkpoint_2.ckpt'
CKPT3='pkl/checkpoint_3.ckpt'
CKPT4='pkl/checkpoint_4.ckpt'
CKPT5='pkl/checkpoint_5.ckpt'

if [ $run_model -eq 1 ]; then
    echo "[2/2] Running ENsiRNA model (using GPU $gpu_id)..."
    
    python run.py \
        --ckpt ${CKPT1} ${CKPT2} ${CKPT3} ${CKPT4} ${CKPT5} \
        --test_set $json_path \
        --save_dir $work_path \
        --gpu $gpu_id \
        --id $mrna_fasta_name
    
    if [ $? -ne 0 ]; then
        echo "Error: Model execution failed"
        exit 1
    fi
    
    echo "✓ Model execution completed"
else
    echo "[2/2] Skipping model execution step"
fi

echo "=== ENsiRNA design pipeline completed successfully ==="
echo "Results saved in: $work_path"
echo "You can find the predicted siRNA efficacy scores in the output directory"