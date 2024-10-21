#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import pickle
import argparse
from typing import List

import numpy as np
import torch

from utils.logger import print_log
from data.mod_utils import MACCSRdkit_VOCAB
########## import your packages below ##########
from tqdm import tqdm
from utils.rna_utils import RNA, VOCAB
from data.mod_utils import MOD_VOCAB
import torch.nn.functional as F
#python -m data.dataset --dataset /public2022/tanwenchong/rna/alldata/train.json --save_dir /public2022/tanwenchong/rna/alldata


phychem={
    't':[322.21,-2.8,4,8,322.05660244,322.05660244,146,21,529,0],
    'c':[323.20,-3.4,5,8,323.05185141,323.05185141,175,21,531,0],
    'g':[363.22,-3.5,6,10,363.05799942,363.05799942,202,24,598,0],
    'a':[347.22,-3.5,5,11,347.06308480,347.06308480,186,23,481,0],
    'u':[363.22,-3.5,6,10,363.05799942,363.05799942,202,24,598,0],
}
sym2idx={
    'a':0,
    'c':1,
    'g':2,
    'u':3,
    't':4,

}


def _generate_data(residues):
    S = []
    PC = []
    mask = torch.ones(25,dtype=torch.long)
    k=0
    for residue in residues:
        S.append(sym2idx[residue])
        PC.append(phychem[residue])
        mask[k] = 0
        k+=1
    PC = torch.tensor(PC,dtype=torch.float)
    S = F.one_hot(torch.tensor(S), num_classes=5)
    Spad = torch.zeros(25 - len(S),5,dtype=torch.long)
    PCpad = torch.zeros(25 - len(S),10,dtype=torch.float)
    S = torch.cat((S,Spad),dim=0)
    PC = torch.cat((PC,PCpad),dim=0)
    return S,PC,mask



def _generate_chain_data(residues):
    S = []
    for residue in residues:
        S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
    S = F.one_hot(torch.tensor(S), num_classes=4)
    pad = torch.zeros(25 - len(S),4,dtype=torch.long)
    S = torch.cat((S,pad),dim=0)
 
    return S


# use this class to splice the dataset and maintain only one part of it in RAM

class E2EDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, save_dir=None, cdr=None,  num_entry_per_file=-1, random=False):

        super().__init__()
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data= []  # list of ABComplex
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except FileNotFoundError:
            print_log('No meta-info file found, start processing', level='INFO')
            need_process = True
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]

        self.ma_vocab,self.ma_dim = MACCSRdkit_VOCAB


    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx
     
    def __len__(self):
        return self.num_entry

    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')

        for line in tqdm(lines):
       
            item = json.loads(line)

            try:
                s,scp,smask = _generate_data(item["sense raw seq"])
                a,acp,amask = _generate_data(item["anti raw seq"])
                S = torch.cat((s,a),dim=0)
                CP = torch.cat((scp,acp),dim=0)
                MASK = torch.cat((smask,amask),dim=0)
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

  
            atom_mask=item['atom_mask']


            
            self.data.append([S,CP,MASK,item['PCT'],atom_mask])

        if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
            self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    ########## override get item ##########
    def __getitem__(self, idx):

        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        #item,PCT,atom_mask= self.data[idx]
        S,CP,MASK,PCT,atom_mask = self.data[idx]
        #item,PCT,atom_mask,rna_pos53,rna_pos35,cls_embeddings = self.data[idx]
       
        
        
        #sense_rna = item.get_chain('A')
        #anti_rna = item.get_chain('B')
        #rna_acids=[]

        #for i in range(len(sense_rna)):
        #    rna_acids.append(sense_rna.get_residue(i))
        #    S = _generate_chain_data(rna_acids)

        #rna_acids=[]
        #for i in range(len(anti_rna)):
        #    rna_acids.append(anti_rna.get_residue(i))
        #    A = _generate_chain_data(rna_acids)

        #S = torch.cat((S,A),dim=0)

        rnamod_embedding=torch.zeros([50,3,self.ma_dim],dtype=torch.long)
     
        for i in range(50):
            k=0
            for j in range(3):  
                if  atom_mask[i][j]   != 0:            
                    rnamod_embedding[i][k] = torch.tensor(self.ma_vocab[atom_mask[i][j]],dtype=torch.long) #self.chem_vocab
                    k+=1
            

        rnamod_embedding = torch.sum(rnamod_embedding,dim=1)
        rnamod_embedding = rnamod_embedding != 0
        rnamod_embedding = rnamod_embedding.float()
    
   
        rna_data = {'S':S ,'CP':CP,'MASK': MASK,'rnamod' : rnamod_embedding}
        rna_data['pct']=[float(PCT)/100]
  
        return rna_data

    @classmethod
    def collate_fn(cls, batch):
   
        keys = ['S','CP', 'MASK','rnamod' ,'pct']
        types = [torch.float,torch.float, torch.bool, torch.float,torch.float]

        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
                res[key] = torch.stack(val)

        return res


def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save processed data')
    return parser.parse_args()
 

if __name__ == '__main__':
    args = parse()
    dataset = E2EDataset(args.dataset, args.save_dir, num_entry_per_file=-1)
    print(len(dataset))