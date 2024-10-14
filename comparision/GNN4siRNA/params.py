##################
# INPUT FILES: path of the three input files
##################

#k-mers of siRNA sequences
sirna_kmer_file = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/dataset_1/sirna_kmers.txt"
#k-mers of mRNA sequences
mrna_kmer_file = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/dataset_1/mRNA_kmers.txt"
#thermodynamic features of siRNA-mRNA interaction
sirna_target_thermo_file = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/dataset_1/sirna_target_thermo.csv"
# sirna_efficacy_values
sirna_efficacy_file = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/dataset_1/sirna_mrna_efficacy.csv"

sirna_efficacy_file2 = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/dataset_2/sirna_mrna_efficacy.csv"
sirna_target_thermo_file2 = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/dataset_2/sirna_target_thermo.csv"
sirna_kmer_file2 = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/dataset_2/sirna_kmers.txt"
mrna_kmer_file2 = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/dataset_2/target_kmers.txt"

sirna_efficacy_file3 = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/patent/sirna_mrna_efficacy.csv"
sirna_target_thermo_file3 = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/patent/sirna_target_thermo.csv"
sirna_kmer_file3 = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/patent/sirna_kmers.txt"
mrna_kmer_file3 = "/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/data/processed/patent/mRNA_kmers.txt"

##################
# GNN Hyperparameters
##################

# specify the minibatch size and the number of epochs for training the GNN model
batch_size = 60
epochs = 100

# two hidden layers HinSAGE sizes
hinsage_layer_sizes = [32, 16]

# sizes of 1- and 2-hop neighbour samples for each hidden layer of the HinSAGE model
hop_samples = [8, 4]

# Dropout value for the HinSAGE model
dropout = 0.15

# Adamax learning rate
lr= 0.001

# loss function
loss='mse'




