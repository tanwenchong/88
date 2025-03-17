# a pipeline for modified siRNA prediction

import argparse
import os
import sys
import torch
import pandas as pd

def parse_arguments():
    """Parse command line arguments for the siRNA prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="ENsiRNA: A pipeline for modified siRNA prediction",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Run in interactive mode with step-by-step prompts")
    parser.add_argument("--id", type=str, help="siRNA ID (e.g., Givosiran)")

    parser.add_argument("-o", "--output-dir", type=str, default="result",
                        help="Directory to save results (default: 'result')")
    parser.add_argument("--model", type=str, default="pkl/checkpoint_1.ckpt",
                        help="Path to model checkpoint (default: pkl/checkpoint_1.ckpt)")
    
    return parser.parse_args()

def input_data_interactive():
    """Collect siRNA data through interactive prompts."""
    print("\n" + "="*80)
    print("Welcome to ENsiRNA: Modified siRNA Prediction Pipeline")
    print("="*80)
    
    print("\nPlease input the following information:")
    
    # Input ID
    print("\n[1/5] Input ID (e.g., Givosiran):")
    ID = input(">> ")

    # Input anti-sense sequence
    print("\n[2/5] Input anti-sense sequence (e.g., UAAGAUGAGACACUCUUUCUGGU):")
    anti_seq = input(">> ")

    # Input anti-sense modification
    anti_mod = {}
    print("\n[3/5] Input anti-sense modifications:")
    print("Format: 'modification_type:positions' (comma-separated positions)")
    print("Examples:")
    print("  2-Fluoro:2,3,4,6,8,10,12,14,16,18,20")
    print("  2-O-Methyl:1,5,7,9,11,13,15,17,19,21,22,23")
    print("  Phosphorothioate:2,3,22,23")
    print("Enter one modification at a time. Type '0' when finished.")
    
    while True:
        mod = input("\nAnti-sense modification (or '0' to finish): ")
        if mod == "0":
            break
        try:
            mod_type, mod_pos = mod.split(":")
            anti_mod[mod_type] = mod_pos
        except ValueError:
            print("Error: Invalid format. Please use 'type:positions' format.")

    # Input sense sequence
    print("\n[4/5] Input sense sequence (e.g., CAGAAAGAGUGUCUCAUCUUA):")
    sense_seq = input(">> ")

    # Input sense modification
    sense_mod = {}
    print("\n[5/5] Input sense modifications:")
    print("Format: 'modification_type:positions' (comma-separated positions)")
    print("Examples:")
    print("  2-Fluoro:2,3,4,6,8,10,12,14,16,18,20")
    print("  2-O-Methyl:1,2,3,4,5,6,8,10,12,14,16,17,18,19,20,21")
    print("  Phosphorothioate:2,3")
    print("Enter one modification at a time. Type '0' when finished.")
    
    while True:
        mod = input("\nSense modification (or '0' to finish): ")
        if mod == "0":
            break
        try:
            mod_type, mod_pos = mod.split(":")
            sense_mod[mod_type] = mod_pos
        except ValueError:
            print("Error: Invalid format. Please use 'type:positions' format.")

    return ID, anti_seq, anti_mod, sense_seq, sense_mod

def process_modifications(mod_dict):
    """Process modification dictionary into position and type strings."""
    if not mod_dict:
        return "", ""
    pos_str = '* '.join([i for i in mod_dict.values()])
    mod_str = ' * '.join([i for i in mod_dict.keys()])
    return mod_str, pos_str

def prepare_data(ID, anti_seq, anti_mod, sense_seq, sense_mod, output_dir):
    """Prepare data and save to Excel file."""
    # Process modifications
    mod_anti, pos_anti = process_modifications(anti_mod)
    mod_sense, pos_sense = process_modifications(sense_mod)

    # Define columns
    columns = [
        'ID', 'source', 'cc', 'sense raw seq', 'sense mod', 'sense pos',
        'anti raw seq', 'anti mod', 'anti pos', 'PCT', 'anti length',
        'sense length', 'cc_norm', 'group'
    ]

    # Create dataframe
    df = pd.DataFrame(columns=columns)
    df.loc[0] = [
        ID, 0, 0, sense_seq, mod_sense, pos_sense, anti_seq, mod_anti, 
        pos_anti, 0, len(anti_seq), len(sense_seq), 0, 0
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Excel
    output_file = f'{output_dir}/{ID}.xlsx'
    df.to_excel(output_file, index=False)
    
    print(f"\n✓ Data prepared successfully and saved to {output_file}")
    return output_file

def generate_pdb_data(excel_file, output_dir):
    """Generate PDB data from Excel file."""

    from data.get_pdb import Data_Prepare
        
    pdb_output_dir = f'{output_dir}/pdb_data'
    os.makedirs(pdb_output_dir, exist_ok=True)
        
    print("\n[1/2] Generating PDB data...")
    data_prepare = Data_Prepare(excel_file, pdb_output_dir)
    data_prepare.process()
    print(f"✓ PDB data generated successfully in {pdb_output_dir}")
    return True


def run_prediction(ID, model_path, output_dir):
    """Run siRNA prediction using the model."""
    try:
        print("\n[2/2] Running prediction...")
        result = os.system(f'bash test.sh {model_path} {output_dir}/{ID}.json {output_dir} {ID}')
        
        if result == 0:
            print(f"✓ Prediction completed successfully. Results saved to {output_dir}/{ID}.xlsx")
            return True
        else:
            print("Error running prediction. Check logs for details.")
            return False
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return False

def parse_modification_args(mod_args):
    """Parse modification arguments from command line."""
    mod_dict = {}
    if mod_args:
        for mod in mod_args:
            try:
                mod_type, mod_pos = mod.split(":")
                mod_dict[mod_type] = mod_pos
            except ValueError:
                print(f"Warning: Ignoring invalid modification format: {mod}")
    return mod_dict

def main():
    """Main function to run the siRNA prediction pipeline."""
    args = parse_arguments()
    
    # Set output directory
    output_dir = os.path.abspath(args.output_dir)
    
    # Interactive or command-line mode

    ID, anti_seq, anti_mod, sense_seq, sense_mod = input_data_interactive()

    
    # Prepare data
    excel_file = prepare_data(ID, anti_seq, anti_mod, sense_seq, sense_mod, output_dir)
    
    # Generate PDB data

    pdb_success = generate_pdb_data(excel_file, output_dir)
    if not pdb_success:
        print("Warning: PDB data generation failed or was incomplete.")

    

    prediction_success = run_prediction(ID, args.model, output_dir)
    if prediction_success:
        print("\n" + "="*80)
        print(f"ENsiRNA pipeline completed successfully for {ID}")
        print(f"Results are available in: {output_dir}")
        print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)