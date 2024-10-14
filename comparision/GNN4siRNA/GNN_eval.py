
"""
Created on Thu Mar 31 13:10:50 2022

@author: fiannaca
"""

print('start')

import stellargraph as StellarGraph
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy
import scipy.sparse
import scipy.sparse.linalg
from sklearn.metrics import roc_auc_score
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
# import file with parameters
import params
import random
import os
# 指定使用的GPU设备
SEED = 12  # 选择一个固定的种子值
StellarGraph.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ["PYTHONHASHSEED"]="0"
print('start')

# Specify the minibatch size and the number of epochs for training the model
batch_size = params.batch_size
epochs = params.epochs


#############################################
# import file with sirna / target thermodynamic features
#############################################

# k-mers of siRNA sequences
sirna_pd = pd.read_csv( params.sirna_kmer_file, header=None )
sirna_pd= sirna_pd.set_index(0)

sirna_dataset2 = pd.read_csv( params.sirna_kmer_file2, header=None )
sirna_dataset2 = sirna_dataset2.set_index(0)

sirna_dataset3 = pd.read_csv( params.sirna_kmer_file3, header=None )
sirna_dataset3 = sirna_dataset3.set_index(0)

# k-mers of mRNA sequences
mRNA_pd = pd.read_csv( params.mrna_kmer_file, header=None )
mRNA_pd= mRNA_pd.set_index(0)

mrna_dataset2 = pd.read_csv( params.mrna_kmer_file2, header=None )
mrna_dataset2 = mrna_dataset2.set_index(0)

mrna_dataset3 = pd.read_csv( params.mrna_kmer_file3, header=None )
mrna_dataset3 = mrna_dataset3.set_index(0)

# thermodynamic features of siRNA-mRNA interaction
thermo_feats_pd = pd.read_csv( params.sirna_target_thermo_file, header=None )
thermo_feats_dataset2 = pd.read_csv( params.sirna_target_thermo_file2, header=None )
thermo_feats_dataset3 = pd.read_csv( params.sirna_target_thermo_file3, header=None )

# sirna_efficacy_values
sirna_efficacy_pd = pd.read_csv( params.sirna_efficacy_file)
sirna_efficacy_pd['efficacy']=sirna_efficacy_pd.groupby('mRNA')['efficacy'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sirna_efficacy_pd['group'] = pd.qcut(sirna_efficacy_pd['efficacy'], 10, labels=False)

sirna_efficacy_dataset2 = pd.read_csv(params.sirna_efficacy_file2)
sirna_efficacy_dataset2['efficacy']=sirna_efficacy_dataset2.groupby('mRNA')['efficacy'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sirna_efficacy_dataset3 = pd.read_csv(params.sirna_efficacy_file3)

# rename first 2 columns in "source" and "target"
thermo_feats_pd.rename(columns={0 : "source", 1 : "target"}, inplace= True)
thermo_feats_dataset2.rename(columns={0 : "source", 1 : "target"}, inplace= True)
thermo_feats_dataset3.rename(columns={0 : "source", 1 : "target"}, inplace= True)

# Here we transform interaction edges in "interaction nodes"
# Intercation node has 2 edges that connect it to siRNA and mRNA, respectively
# Node ID cames from source and target ids
interaction_pd = thermo_feats_pd.drop(['source','target'], axis=1)
interaction_pd["index"] = thermo_feats_pd['source'].astype(str) + "_" + thermo_feats_pd['target']
interaction_pd= interaction_pd.set_index("index")

interaction_dataset2 = thermo_feats_dataset2.drop(['source','target'], axis=1)
interaction_dataset2["index"] = thermo_feats_dataset2['source'].astype(str) + "_" + thermo_feats_dataset2['target']
interaction_dataset2= interaction_dataset2.set_index("index")

interaction_dataset3 = thermo_feats_dataset3.drop(['source','target'], axis=1)
interaction_dataset3["index"] = thermo_feats_dataset3['source'].astype(str) + "_" + thermo_feats_dataset3['target']
interaction_dataset3= interaction_dataset3.set_index("index")

# New edges have no features
sirna_edge_pd_no_feats = thermo_feats_pd[['source','target']]
data1 = { 'source' : list(interaction_pd.index),
         'target' : sirna_edge_pd_no_feats['source']}
data2 = { 'source' : list(interaction_pd.index),
         'target' : sirna_edge_pd_no_feats['target']}

sirna_edge_pd_no_feats_dataset2 = thermo_feats_dataset2[['source','target']]
data1_dataset2 = { 'source' : list(interaction_dataset2.index),
         'target' : sirna_edge_pd_no_feats_dataset2['source']}
data2_dataset2 = { 'source' : list(interaction_dataset2.index),
         'target' : sirna_edge_pd_no_feats_dataset2['target']}

sirna_edge_pd_no_feats_dataset3 = thermo_feats_dataset3[['source','target']]
data1_dataset3 = { 'source' : list(interaction_dataset3.index),
         'target' : sirna_edge_pd_no_feats_dataset3['source']}
data2_dataset3 = { 'source' : list(interaction_dataset3.index),
         'target' : sirna_edge_pd_no_feats_dataset3['target']}


all_my_edges = pd.DataFrame(data1)
all_my_edges_temp = pd.DataFrame(data2)

all_my_edges_dataset2 = pd.DataFrame(data1_dataset2)
all_my_edges_temp_dataset2 = pd.DataFrame(data2_dataset2)

all_my_edges_dataset3 = pd.DataFrame(data1_dataset3)
all_my_edges_temp_dataset3 = pd.DataFrame(data2_dataset3)

# Merge all the edges 
all_my_edges = pd.concat([all_my_edges, all_my_edges_temp], ignore_index = True, axis = 0)
all_my_edges_dataset2 = pd.concat([all_my_edges_dataset2, all_my_edges_temp_dataset2], ignore_index = True, axis = 0)
all_my_edges_dataset3 = pd.concat([all_my_edges_dataset3, all_my_edges_temp_dataset3], ignore_index = True, axis = 0)


# We want to predict the interaction weight, i.e. the label of interaction node
interaction_weight = sirna_efficacy_pd['efficacy']
interaction_weight = interaction_weight.set_axis(interaction_pd.index)


interaction_weight_dataset2 = sirna_efficacy_dataset2['efficacy']
interaction_weight_dataset2 = interaction_weight_dataset2.set_axis(interaction_dataset2.index)

interaction_weight_dataset3 = sirna_efficacy_dataset3['efficacy']
interaction_weight_dataset3 = interaction_weight_dataset3.set_axis(interaction_dataset3.index)

# Create Stellargraph object
my_stellar_graph = StellarGraph.StellarGraph( {"siRNA": sirna_pd, "mRNA": mRNA_pd, "interaction": interaction_pd}, 
                                             edges=all_my_edges, source_column="source", target_column= "target") 

my_stellar_graph_dataset2 = StellarGraph.StellarGraph( {"siRNA": sirna_dataset2, "mRNA": mrna_dataset2, "interaction": interaction_dataset2}, 
                                             edges=all_my_edges_dataset2, source_column="source", target_column= "target")

my_stellar_graph_dataset3 = StellarGraph.StellarGraph( {"siRNA": sirna_dataset3, "mRNA": mrna_dataset3, "interaction": interaction_dataset3}, 
                                             edges=all_my_edges_dataset3, source_column="source", target_column= "target")

print('data finish')
################################################
# Create the model
################################################

overall_PCC = []
overall_mse = []

# HinSAGE parameters
hinsage_layer_sizes = params.hinsage_layer_sizes
hop_samples = params.hop_samples
dropout = params.dropout
loss_function= params.loss
learning_rate= params.lr

result=np.zeros([5,10])

# with range = 1, it make only a repeat
with tf.device('/GPU:0'):
    score_PCC1 = []
    score_PCC2 = []
    score_PCC3 = []
    score_SCC1 = []
    score_SCC2 = []
    score_SCC3 = []
    score_mse1= []
    score_mse2 = []
    score_mse3 = []
    score_mae1 = []
    score_mae2 = []
    score_mae3 = []
    score_auc1 = []
    score_auc2 = []
    score_auc3 = []



    
    # perform 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) #42
    df_test = pd.DataFrame() 
    df_patent = pd.DataFrame()
    for fold, (train, test) in enumerate(skf.split(sirna_efficacy_pd, sirna_efficacy_pd['group'])):    
        
    #kfold = KFold(n_splits=10, shuffle=True, random_state = 2)
    #for train, test in kfold.split(interaction_weight):
        train_interaction, test_interaction = interaction_weight[train], interaction_weight[test]

        #print("-- Fold n. ", fold)

        
# Create the generators to feed data from the graph to the Keras model
# We specify we want to make node regression on the "interaction" node
        generator = HinSAGENodeGenerator(
            my_stellar_graph, batch_size, hop_samples, head_node_type= "interaction",seed=SEED)

        generator_dataset2 = HinSAGENodeGenerator(
            my_stellar_graph_dataset2, batch_size, hop_samples, head_node_type= "interaction",seed=SEED)

        generator_dataset3 = HinSAGENodeGenerator(
            my_stellar_graph_dataset3, batch_size, hop_samples, head_node_type= "interaction",seed=SEED)

        train_gen = generator.flow(train_interaction.index, train_interaction, shuffle=True)

        hinsage_model = HinSAGE(
            layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=dropout
            )
        
# Expose input and output sockets of hinsage:
        x_inp, x_out = hinsage_model.in_out_tensors()

        prediction = layers.Dense(units=1)(x_out)


# Now let’s create the actual Keras model with the graph inputs x_inp 
# provided by the graph_model and outputs being the predictions
        model = Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=optimizers.Adam(lr=learning_rate),
            loss=loss_function
            )


# Train the model, keeping track of its loss and accuracy on the training set, 
# and its generalisation performance on the test set

        test_gen = generator.flow(test_interaction.index, test_interaction)

        test_gen_dataset2 = generator_dataset2.flow(interaction_weight_dataset2.index, interaction_weight_dataset2)

        test_gen_dataset3 = generator_dataset3.flow(interaction_weight_dataset3.index, interaction_weight_dataset3)

        history = model.fit(
            train_gen, epochs=epochs, validation_data=test_gen, verbose=2, shuffle=False,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
            workers=1, use_multiprocessing=False
            )


# Plot the training history, epochs Vs loss
#        StellarGraph.utils.plot_history(history)


# Now we have trained the model we can evaluate on the test set.
        pred1 = model.predict(test_gen,workers=1, use_multiprocessing=False)
        pred2 = model.predict(test_gen_dataset2,workers=1, use_multiprocessing=False)
        pred3 = model.predict(test_gen_dataset3,workers=1, use_multiprocessing=False)
        
        df_test[f'result_{fold}'] = [i[0] for i in pred2]
        df_patent[f'result_{fold}'] = [i[0] for i in pred3]
       

        pearson1 = scipy.stats.pearsonr(test_interaction.values,[i[0] for i in pred1])
        pearson2 = scipy.stats.pearsonr(interaction_weight_dataset2.values,[i[0] for i in pred2])
        pearson3 = scipy.stats.pearsonr(interaction_weight_dataset3.values,[i[0] for i in pred3])
        spearman1 = scipy.stats.spearmanr(test_interaction.values,[i[0] for i in pred1])
        spearman2 = scipy.stats.spearmanr(interaction_weight_dataset2.values,[i[0] for i in pred2])
        spearman3 = scipy.stats.spearmanr(interaction_weight_dataset3.values,[i[0] for i in pred3])
        score_PCC1.append(pearson1[0])
        score_PCC2.append(pearson2[0])
        score_PCC3.append(pearson3[0])
        score_SCC1.append(spearman1[0])
        score_SCC2.append(spearman2[0])
        score_SCC3.append(spearman3[0])
        mse_run1 = mean_squared_error(test_interaction.values,[i[0] for i in pred1])    
        mse_run2 = mean_squared_error(interaction_weight_dataset2.values,[i[0] for i in pred2])
        mse_run3 = mean_squared_error(interaction_weight_dataset3.values,[i[0] for i in pred3])
        score_mse1.append(mse_run1)
        score_mse2.append(mse_run2)
        score_mse3.append(mse_run3)
        mae_run1 = mean_absolute_error(test_interaction.values,[i[0] for i in pred1])
        mae_run2 = mean_absolute_error(interaction_weight_dataset2.values,[i[0] for i in pred2])
        mae_run3 = mean_absolute_error(interaction_weight_dataset3.values,[i[0] for i in pred3])
        score_mae1.append(mae_run1)
        score_mae2.append(mae_run2)
        score_mae3.append(mae_run3)
        AUC1 = roc_auc_score(test_interaction.values>0.5,[i[0] for i in pred1])
        AUC2 = roc_auc_score(interaction_weight_dataset2.values>0.5,[i[0] for i in pred2])
        AUC3 = roc_auc_score(interaction_weight_dataset3.values>0.5,[i[0] for i in pred3])
        score_auc1.append(AUC1)
        score_auc2.append(AUC2)
        score_auc3.append(AUC3)
    
    df_test.to_csv(f'/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/paper_result/result_test.csv', index=False)
    df_patent.to_csv(f'/public2022/tanwenchong/GNN/gnn4sirna/GNN4siRNA-main/paper_result/result_patent.csv', index=False)
    
    print('valid PCC1',f'{np.mean(score_PCC1):.4f}',','.join([f'{i:.4f}' for i in score_PCC1]))
    print('valid PCC2',f'{np.mean(score_PCC2):.4f}',','.join([f'{i:.4f}' for i in score_PCC2]))
    print('valid PCC3',f'{np.mean(score_PCC3):.4f}',','.join([f'{i:.4f}' for i in score_PCC3]))
    print('valid SCC1',f'{np.mean(score_SCC1):.4f}',','.join([f'{i:.4f}' for i in score_SCC1]))
    print('valid SCC2',f'{np.mean(score_SCC2):.4f}',','.join([f'{i:.4f}' for i in score_SCC2]))
    print('valid SCC3',f'{np.mean(score_SCC3):.4f}',','.join([f'{i:.4f}' for i in score_SCC3]))
    print('valid MSE1',f'{np.mean(score_mse1):.4f}',','.join([f'{i:.4f}' for i in score_mse1]))
    print('valid MSE2',f'{np.mean(score_mse2):.4f}',','.join([f'{i:.4f}' for i in score_mse2]))
    print('valid MSE3',f'{np.mean(score_mse3):.4f}',','.join([f'{i:.4f}' for i in score_mse3]))
    print('valid MAE1',f'{np.mean(score_mae1):.4f}',','.join([f'{i:.4f}' for i in score_mae1]))
    print('valid MAE2',f'{np.mean(score_mae2):.4f}',','.join([f'{i:.4f}' for i in score_mae2]))
    print('valid MAE3',f'{np.mean(score_mae3):.4f}',','.join([f'{i:.4f}' for i in score_mae3]))
    print('valid AUC1',f'{np.mean(score_auc1):.4f}',','.join([f'{i:.4f}' for i in score_auc1]))
    print('valid AUC2',f'{np.mean(score_auc2):.4f}',','.join([f'{i:.4f}' for i in score_auc2]))
    print('valid AUC3',f'{np.mean(score_auc3):.4f}',','.join([f'{i:.4f}' for i in score_auc3]))

    
