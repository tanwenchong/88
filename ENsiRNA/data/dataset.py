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
from utils.rna_utils import SIRNA, VOCAB ,RNAFeature

import RNA
#python -m data.dataset --dataset /public2022/tanwenchong/rna/alldata/train.json --save_dir /public2022/tanwenchong/rna/alldata
import fm
device = 'cuda'
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
model.to(device=device)
#mmodel, malphabet = fm.pretrained.mrna_fm_t12()
#mbatch_converter = malphabet.get_batch_converter()
#mmodel.eval()  # disables dropout for deterministic results
#mmodel.to(device=device)

en_dict = {
    'AA':-0.93,
    'UU':-0.93,
    'AU':-1.10,
    'UA':-1.33,
    'CU':-2.08,
    'AG':-2.08,
    'CA':-2.11,
    'UG':-2.11,
    'GU':-2.24,
    'AC':-2.24,          
    'GA':-2.35,
    'UC':-2.35,
    'CG':-2.36,
    'GG':-3.26,
    'CC':-3.26,
    'GC':-3.42,
}

def _generate_data(residues,chain):

    energys = []
    if chain == 'mrna':
        if residues[0] == '-':
            mean_energy = en_dict[residues[1:3]]
        else:
            mean_energy = (en_dict[residues[0:2]] + en_dict[residues[1:3]]) / 2
        energys.append(mean_energy)

        for i in range(2,len(residues)-2):
            mean_energy = (en_dict[residues[i-1:i+1]]+en_dict[residues[i:i+2]]) / 2
            energys.append(mean_energy)
        if residues[-1] == '-':
            mean_energy = en_dict[residues[-3:-1]]
        else:
            mean_energy = (en_dict[residues[-3:-1]] + en_dict[residues[-2:]]) / 2
        energys.append(mean_energy)
    if chain == 'sirna':
        energys.append(en_dict[residues[0:2]])
        for i in range(1,len(residues)-1):
            mean_energy = (en_dict[residues[i-1:i+1]]+en_dict[residues[i:i+2]]) / 2
            energys.append(mean_energy)
        energys.append(en_dict[residues[-2:]])
    #print(len(energys),len(residues),chain)
    return energys
    




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

def single_energy(sequence,left_padlen=0,right_padlen=0):
    sequence = sequence.replace('\n','').replace('T','U')
    # 计算整个序列的自由能
    fc = RNA.fold_compound(sequence)
    (ss, mfe) = fc.mfe()
    base_contributions = [mfe]
    if left_padlen > 0:
        base_contributions += [0] * left_padlen
    for i in range(len(sequence)):
        # 创建一个新的序列,将第i个碱基替换为'X'
        mutated_seq = sequence[:i] + 'X' + sequence[i+1:]
        
        # 计算突变序列的自由能
        fc_mut = RNA.fold_compound(mutated_seq)
        (_, mfe_mut) = fc_mut.mfe()
        
        # 计算该碱基的贡献
        contribution = mfe_mut - mfe
        base_contributions.append(-contribution)
    if right_padlen > 0:
        base_contributions += [0] * right_padlen
    return base_contributions
def calculate_duplex_energy(seq1, seq2):
    # 计算双链RNA的自由能
    duplexes = RNA.duplexfold(seq1, seq2)
    return duplexes.energy
def duplex_energy(seq1, seq2,left_padlen=0,right_padlen=0):
    # 计算整个双链的自由能
    total_energy = calculate_duplex_energy(seq1, seq2)
    
    contributions = [total_energy]
    if left_padlen > 0:
        contributions += [0] * left_padlen
    # 计算第一条链中每个碱基的贡献
    for i in range(len(seq1)):
        mutated_seq1 = seq1[:i] + 'X' + seq1[i+1:]
        mutated_energy = calculate_duplex_energy(mutated_seq1, seq2)
        contribution = mutated_energy - total_energy
        contributions.append(-contribution)
    if right_padlen > 0:
        contributions += [0] * right_padlen
    # 计算第二条链中每个碱基的贡献
    for i in range(len(seq2)):
        mutated_seq2 = seq2[:i] + 'X' + seq2[i+1:]
        mutated_energy = calculate_duplex_energy(seq1, mutated_seq2)
        contribution = mutated_energy - total_energy
        contributions.append(-contribution)
    
    return contributions

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
                cplx = SIRNA.from_pdb(
                    item['pdb_data_path'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            sec_pos = item['start']
            chain = item['chain']
            
            padlen = int((61 - len(item['anti seq'])) / 2)
            rna_pos= [0]
            for i in range(51,51+61):
                rna_pos.append(i)
            rna_pos.append(1)
            for i in range(2,2+len(item['sense seq'])):
                rna_pos.append(i)
            for i in range(26,26+len(item['anti seq'])):
                rna_pos.append(i)
            
            fm_mseq = ''
            real_mrna = ''
            padlen = int((61 - len(item['anti seq'])) / 2)
            position = int(item['position'])
            anti_seq_len = len(item['anti seq'])
            mRNA_seq_len = len(item['mRNA_seq'])
            left_padlen = 0
            right_padlen = 0
            # 左侧填充
            if position - padlen < 0:
                left_padlen = padlen - position
                fm_mseq = '-' * left_padlen
                fm_mseq += item['mRNA_seq'][:position]
                
            else:
                fm_mseq = item['mRNA_seq'][position - padlen:position]
                
           
            # 中间序列
            fm_mseq += item['mRNA_seq'][position:position + anti_seq_len]
          

            # 右侧填充
            if position + anti_seq_len + padlen > mRNA_seq_len:
                right_padlen = position + anti_seq_len + padlen - mRNA_seq_len
                fm_mseq += item['mRNA_seq'][position + anti_seq_len:]
                fm_mseq += '-' * right_padlen
            
            else:
                fm_mseq += item['mRNA_seq'][position + anti_seq_len:position + anti_seq_len + padlen]
               
            real_mrna = fm_mseq.upper().replace('T','U').replace('N','X').replace('-','')
            fm_mseq = fm_mseq.upper().replace('T','U').replace('N','X')
            mseq = fm_mseq.replace('-','#')  # 将序列转换为大写字母 seq with pad for model
            

            mrna_anti_seq = [("mrna_anti", real_mrna + item['anti seq'].upper())]
            sense_anti_seq = [("sense_anti", item['sense seq'].upper()+item['anti seq'].upper())]
            mrna_seq = [("mrna", real_mrna)]
            sense_seq = [("sense", item['sense seq'].upper())]
            anti_seq = [("anti", item['anti seq'].upper())]



            
            sense_anti_energy = duplex_energy(item['sense seq'].upper(),item['anti seq'].upper())
            mrna_anti_energy = duplex_energy(real_mrna,item['anti seq'].upper(),left_padlen,right_padlen)
            mrna_sense_energy = duplex_energy(real_mrna,item['sense seq'].upper(),left_padlen,right_padlen)
            mrna_energys = single_energy(real_mrna,left_padlen,right_padlen)
            anti_energys = single_energy(item['anti seq'].upper())
            sense_energys = single_energy(item['sense seq'].upper())
            #single chain
            single_energys = mrna_energys + [anti_energys[0]] + sense_energys[1:] + anti_energys[1:]
            #cross duplex
            #duplex1_energys = mrna_anti_energy[:1+len(fm_mseq)] + mrna_sense_energy[:1+len(item['sense seq'])] + mrna_anti_energy[1+len(fm_mseq):]
            #sirna duplex
            #duplex2_energys = mrna_energys + sense_anti_energy 
            #duplex_energys = mrna_anti_energy[:1+len(fm_mseq)] + sense_anti_energy[:1+len(item['sense seq'])] + mrna_anti_energy[1+len(fm_mseq):]
            duplex1_energys = mrna_anti_energy[:1+len(fm_mseq)] + sense_anti_energy[:1+len(item['sense seq'])] + mrna_anti_energy[1+len(fm_mseq):]
            duplex2_energys = mrna_energys + sense_anti_energy 
            energys = np.stack([single_energys, duplex1_energys, duplex2_energys], axis=-1)
            #energys = np.stack([single_energys, duplex_energys], axis=-1)
          

            
            # sense_anti_seq
            batch_labels, batch_strs, batch_tokens = batch_converter(sense_anti_seq)
            with torch.no_grad():
                results = model(batch_tokens.to(device=device), repr_layers=[12])
                sense_anti_embeddings = results["representations"][12][0][:-1].cpu()
            # mrna_anti_seq
            batch_labels, batch_strs, batch_tokens = batch_converter(mrna_anti_seq)
            with torch.no_grad():
                results = model(batch_tokens.to(device=device), repr_layers=[12])
                mrna_anti_embeddings = results["representations"][12][0][:-1].cpu()
            if left_padlen > 0:
                mrna_anti_embeddings = torch.cat([torch.zeros(left_padlen,mrna_anti_embeddings.shape[1]),mrna_anti_embeddings],dim=0)
            if right_padlen > 0:
                mid_embeddings = torch.cat([mrna_anti_embeddings[:1+len(real_mrna)+left_padlen],torch.zeros(right_padlen,mrna_anti_embeddings.shape[1])],dim=0)
                mrna_anti_embeddings = torch.cat([mid_embeddings,mrna_anti_embeddings[1+len(real_mrna)+left_padlen:]],dim=0)
            # mrna_seq
            batch_labels, batch_strs, batch_tokens = batch_converter(mrna_seq)
            with torch.no_grad():
                results = model(batch_tokens.to(device=device), repr_layers=[12])
                mrna_embeddings = results["representations"][12][0][:-1].cpu()
            if left_padlen > 0:
                mrna_embeddings = torch.cat([torch.zeros(left_padlen,mrna_embeddings.shape[1]),mrna_embeddings],dim=0)
            if right_padlen > 0:
                mrna_embeddings = torch.cat([mrna_embeddings,torch.zeros(right_padlen,mrna_embeddings.shape[1])],dim=0)
            # sense_seq
            batch_labels, batch_strs, batch_tokens = batch_converter(sense_seq)
            with torch.no_grad():
                results = model(batch_tokens.to(device=device), repr_layers=[12])
                sense_embeddings = results["representations"][12][0][:-1].cpu()
            # anti_seq
            batch_labels, batch_strs, batch_tokens = batch_converter(anti_seq)
            with torch.no_grad():
                results = model(batch_tokens.to(device=device), repr_layers=[12])
                anti_embeddings = results["representations"][12][0][:-1].cpu()

            single_embeddings = torch.cat([mrna_embeddings,anti_embeddings[0].unsqueeze(0),sense_embeddings[1:],anti_embeddings[1:]],dim=0)
            duplex_embeddings = torch.cat([mrna_anti_embeddings[:1+len(fm_mseq)],sense_anti_embeddings[:1+len(item['sense seq'])],mrna_anti_embeddings[1+len(fm_mseq):]],dim=0)



    
                

            self.data.append([cplx,item['efficacy'],sec_pos,rna_pos,single_embeddings,duplex_embeddings,energys,mseq,chain])
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
    
        item,PCT,sec_pos,rna_pos,single_embeddings,duplex_embeddings,energys,mseq,chain = self.data[idx]
     
        #print(mseq)
        mseq = mseq.replace('#','N').replace('R','N').replace('X','N')
        
        sense_rna = item.get_chain('A')
        anti_rna = item.get_chain('B')
        rna_acids=[]

        for i in range(len(sense_rna)):
            rna_acids.append(sense_rna.get_residue(i))
        
        for i in range(len(anti_rna)):
            rna_acids.append(anti_rna.get_residue(i))

        rna_data = _generate_chain_data(rna_acids, VOCAB.BOS)
        
        #print(len(mseq),mseq)
        rna_data['S'] = [VOCAB.symbol_to_idx(_) for _ in '-'+mseq] + rna_data['S']

        rna_data['X'] = np.concatenate([
            np.zeros((len(mseq)+1, 3,3)),
            np.array(rna_data['X']),
        ])
        if len(rna_data['S']) != len(rna_pos):
            print("len(rna_data['S']) != len(rna_pos)")
        smask = [0 for _ in range(len(rna_data['S']))]
        smask[0]=1
        smask[len(mseq)+1]=1
   
        rna_data['energys']=energys 
        rna_data['smask']=smask
        rna_data['rna_pos'] = rna_pos
        rna_data['sec_pos'] = sec_pos
        rna_data['chain'] = chain
        rna_data['pct']=[float(PCT)]
        rna_data['single_embeddings'] = single_embeddings.tolist()
        rna_data['duplex_embeddings'] = duplex_embeddings.tolist()

  
        return rna_data

    @classmethod
    def collate_fn(cls, batch):
   

        keys = ['X', 'S', 'rna_pos','sec_pos', 'pct','smask','single_embeddings','duplex_embeddings','chain','energys']


        types = [torch.float, torch.long, torch.long,torch.long, torch.float,torch.bool,torch.float,torch.float,torch.long,torch.float]

        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                try:
                    val.append(torch.tensor(item[key], dtype=_type))
                except:
                    print(key,item[key])
            
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
