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

########## import your packages below ##########
from tqdm import tqdm
from utils.rna_utils import RNA, VOCAB ,RNAFeature
from data.mod_utils import MOD_VOCAB
from data.mod_utils import MOGANRdkit_VOCAB#,ChemUnimol_VOCAB
#python -m data.dataset --dataset /public2022/tanwenchong/rna/alldata/train.json --save_dir /public2022/tanwenchong/rna/alldata
import fm
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
model.to(device='cuda')

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




def _generate_chain_data(residues, start):
    backbone_atoms = VOCAB.backbone_atoms
    # Coords, Sequence, residue positions, mask for loss calculation (exclude missing coordinates)
    X, S, res_pos,  = [], [], []
    # global node
    # coordinates will be set to the center of the chain
    X.append([[0, 0, 0] for _ in range(VOCAB.MAX_ATOM_NUMBER)])  
    S.append(VOCAB.symbol_to_idx(VOCAB.BOS))
    res_pos.append(0)

    # other nodes
    for residue in residues:
        residue_xloss_mask = [0 for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        bb_atom_coord = residue.get_backbone_coord_map()
        sc_atom_coord = residue.get_sidechain_coord_map()
        if 'P' not in bb_atom_coord:
            for atom in bb_atom_coord:
                ca_x = bb_atom_coord[atom]
                print_log(f'no ca, use {atom}', level='DEBUG')
                break
        else:
            ca_x = bb_atom_coord['P']
        x = [ca_x for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        
        i = 0
        for atom in backbone_atoms:
            if atom in bb_atom_coord:
                x[i] = bb_atom_coord[atom]
            i += 1
        for atom in residue.sidechain:
            if atom in sc_atom_coord:
                x[i] = sc_atom_coord[atom]
            i += 1

        X.append(x)
        S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
        res_pos.append(residue.get_id()[0])
 
    X = np.array(X)
    center = np.mean(X[1:].reshape(-1, 3), axis=0)
    X[0] = center  # set center
    
    data = {'X': X, 'S': S} #'residue_pos': res_pos
    #data = {'X': X, 'S': S ,'residue_pos': res_pos} #'residue_pos': res_pos
    return data


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
        self.mg_vocab,self.mg_dim = MOGANRdkit_VOCAB #ChemUnimol_VOCAB


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
        # line_id = 0
        self.mg_vocab,self.mg_dim = MOGANRdkit_VOCAB
        for line in tqdm(lines):
            # if line_id < 206:
            #     line_id += 1
            #     continue
            item = json.loads(line)

            try:
                cplx = RNA.from_pdb(
                    item['pdb_data_path'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            RNA_OH = RNAFeature.seq_to_raw(item['sense raw seq'],item['anti raw seq'])
            RNA_PC = RNAFeature.seq_to_phychem(item['sense raw seq'],item['anti raw seq'])
  
       
            sec_pos = item['start']
            chain_id = [0]
            


            rna_pos= [0]
            for i in range(1,1+len(item['sense seq'])):
                rna_pos.append(i)
            for i in range(26,26+len(item['anti seq'])):
                rna_pos.append(i)

            marker = 0 # item['marker']

            fm_seq = [("RNA1", item['sense seq'].upper()+item['anti seq'].upper())]
            batch_labels, batch_strs, batch_tokens = batch_converter(fm_seq)
            with torch.no_grad():
                results = model(batch_tokens.to(device='cuda'), repr_layers=[12])
                cls_embeddings = results["representations"][12][0][:-1].cpu()

            rna_len = 1+len(item['sense seq'])+len(item['anti seq'])

            mod_mask = np.zeros(rna_len)
            mod_mask[0] = 1
            mod_mask[item['smask']] = 1

            atom_mask = item['atom_mask']
            rna_mod=np.zeros([rna_len ,3,512])
            #print(PCT,cc)
            
            for i in range(rna_len):
                k=0
                if i!=0:
                    for j in range(3):
                        try:
                            if  atom_mask[i][j]   != 0:
                                rna_mod[i][k] = self.mg_vocab[atom_mask[i][j]]
                                k+=1
                        except Exception as e:
                            print(e)
            rna_mod[0] = np.mean(rna_mod[1:])
 

                


            
            #self.data.append([cplx,item['PCT'],atom_mask,rna_pos53,rna_pos35])
            self.data.append([cplx,item['PCT'],marker,item['atom_mask'],sec_pos,RNA_OH,chain_id,rna_pos,item['cc'],cls_embeddings,rna_mod,mod_mask])
            #self.data.append([cplx,item['PCT'],atom_mask])
        if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
            self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    ########## override get item ##########
    def __getitem__(self, idx):

        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        #item,PCT,atom_mask= self.data[idx]
    
        item,PCT,marker,atom_mask,sec_pos,RNA_OH , chain_id,rna_pos,cc ,cls_embeddings,rna_mod,mod_mask = self.data[idx]
        #item,PCT,atom_mask,rna_pos53,rna_pos35,cls_embeddings = self.data[idx]
       
        
        
        sense_rna = item.get_chain('A')
        anti_rna = item.get_chain('B')
        rna_acids=[]

        for i in range(len(sense_rna)):
            rna_acids.append(sense_rna.get_residue(i))
        for i in range(len(anti_rna)):
            rna_acids.append(anti_rna.get_residue(i))

        rna_data = _generate_chain_data(rna_acids, VOCAB.BOS)
        if len(rna_data['S']) != len(rna_pos):
            print(PCT,cc)

        smask = [0 for _ in range(len(rna_data['S']))]
        smask[0]=2




        rna_data['smask']=smask
        rna_data['atom_mask']=atom_mask
        rna_data['rna_pos'] = rna_pos
        rna_data['sec_pos'] = sec_pos
        rna_data['rna_raw'] = RNA_OH
        #rna_data['rna_pp'] = RNA_PC
        rna_data['chain_id'] = chain_id
        rna_data['mod'] = rna_mod
        rna_data['mod_mask'] = mod_mask
        rna_data['marker'] = marker
        rna_data['cc'] = [float(cc)]

        #rna_data['rna_pos35'] = rna_pos35
 
        rna_data['pct']=[float(PCT)/100]
        rna_data['marker']=[float(marker)/100]
        rna_data['FM'] = cls_embeddings.tolist()


  
        return rna_data

    @classmethod
    def collate_fn(cls, batch):
   
        #keys = ['X', 'S', 'rna_pos', 'pct','gt','smask','atom_mask','FM']
        keys = ['X', 'S', 'rna_pos','sec_pos', 'pct','marker','smask','atom_mask','mod_mask','rna_raw','chain_id','mod','cc','FM']

        #types = [torch.float, torch.long, torch.long, torch.float,torch.float,torch.bool,torch.long,torch.float]
        types = [torch.float, torch.long, torch.long,torch.long, torch.float,torch.float,torch.bool,torch.long,torch.long,torch.long,torch.long,torch.float,torch.float,torch.float]

        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
                res[key] = torch.cat(val, dim=0)
            
        lengths = [len(item['S']) for item in batch]
        
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
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
