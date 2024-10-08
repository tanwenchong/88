#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm
from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from data.dataset import E2EDataset
from utils.rna_utils import VOCAB
from utils.logger import print_log
from utils.random_seed import setup_seed

from utils.logger import print_log
from sklearn.metrics import roc_auc_score,mean_squared_error, mean_absolute_error, r2_score ,roc_auc_score,average_precision_score
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.manifold import TSNE
from matplotlib import markers #41
colors = dict(mcolors.CSS4_COLORS)
markers = list(markers.MarkerStyle.markers.keys())
color_names = list(colors.keys())[:51]

def generate(args):

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # model_type
    #print_log(f'Model type: {type(model)}')



    for i in args.test_set:

        # load test set
        test_set = E2EDataset(i)
        test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=E2EDataset.collate_fn)


        # create save dir
        if args.save_dir is None:
            save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
        else:
            save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ID=[]
        with open(i, 'r') as fin:
            lines = fin.read().strip().split('\n')
            for line in tqdm(lines):
        
                item = json.loads(line)

                ID.append(item['ID'])
        
        df_out=pd.DataFrame()
        summary_items = []
        probs3=[]
        probs1=[]
        probs2=[]
        pcts=[]
        markers=[]
        pos=[]
        emb=[]
        rna=[]
        
        for batch in tqdm(test_loader):
            pcts+=batch['pct'].tolist()
            markers+=batch['marker'].tolist()
            pos+=batch['rna_pos'].tolist()
            rna+=batch['rna_raw'].tolist()

            with torch.no_grad():
                # move data
                for k in batch:
                    if hasattr(batch[k], 'to'):
                        batch[k] = batch[k].to(device)
                
                prob2,h = model.test(**batch)
        

                #prob = model.test(**batch)
                #probs3 += prob3.cpu().tolist()
                #probs1 += prob1.cpu().tolist()
                probs2 += prob2.cpu().tolist()
                #emb += h.cpu().tolist()


        #df_out['ID'] = ID
        #df_out['result'] = probs
        #df_out.to_excel(f'{save_dir}/result.xlsx')

        #print('gts',gts)

        print('pred mod\n',','.join([f'{i:.4f}' for i in probs2]))



#mod
    probs = np.array(probs2) #3
    markers = np.array(markers)
    pcts = np.array(pcts)
    gts = pcts

    
    mse = mean_squared_error(gts,probs)
    mae = mean_absolute_error(gts,probs)
    r2 = r2_score(gts,probs)
    print(f'mse: {mse}')
    print(f'mae: {mae}')
    print(f'r2: {r2}')

  
    #print(f'0.5roc_auc: {roc_auc_score(gts>0.5,probs)}')
    print(f'0.5auprc: {average_precision_score(gts>0.5,probs)}')
    print(f'0.6roc_auc: {roc_auc_score(gts>0.6,probs)}')
    print(f'0.6auprc: {average_precision_score(gts>0.6,probs)}')
    print(f'0.7roc_auc: {roc_auc_score(gts>0.7,probs)}')
    print(f'0.7auprc: {average_precision_score(gts>0.7,probs)}')

    pearson_corr, _ = pearsonr(gts,probs)
    print(f"mod Pearson correlation coefficient: {pearson_corr}")
    spearman_corr, _ = spearmanr(gts,probs)
    print(f"mod Spearman's rank correlation coefficient: {spearman_corr}")
    try:
        print(f'roc_auc_score: {roc_auc_score(auc_truth, np.array(probs))}')
    except:
        print('wrong')

    plot = False
    if plot ==True:
    
    
        tsne = TSNE(n_components=2, random_state=42)
        emb=np.array(emb)
        labels=np.array(pos)
        #labels = auc_truth
        #labels=np.array(rna)
        labels = (labels < 12) | (labels > 38)
        embedded_data = tsne.fit_transform(emb)

        plt.figure(figsize=(20, 20))
        for label in np.unique(labels):

            plt.scatter(embedded_data[labels == label, 0], embedded_data[labels == label, 1], label=f'{label}',c=color_names[int(label)],marker=markers[int(label)//2])
            
        plt.title('t-SNE Visualization')
        plt.legend()
        plt.savefig('/public2022/tanwenchong/rna/fig/node_3d.png')







def parse():
    parser = argparse.ArgumentParser(description='Generate antibodies given epitopes')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_set', type=str, required=True,nargs='+', help='Path to test set')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(12)
    generate(parse())
