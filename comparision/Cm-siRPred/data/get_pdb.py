import pandas as pd

import json
from utils.rna_utils import VOCAB
import argparse
import os
from data.mod_utils import MOD_VOCAB
import torch
import RNA
import re
import subprocess
import multiprocessing


#example （5'->3' and 3'->5'）
#seq1 = "CUUACGCUGAGUACUUCGA".lower()
#seq2 = "GAAUGCGACUCAUGAAGCU".lower()[::-1]



#python -m data.get_pdb 
RF='/public2022/tanwenchong/app/rosetta/rosetta.binary.linux.release-371'
FF='/public2022/tanwenchong/app/rosetta/rosetta.binary.linux.release-371/main/source/bin/rna_denovo.static.linuxgccrelease'
EX='/public2022/tanwenchong/app/rosetta/rosetta.binary.linux.release-371/main/tools/rna_tools/silent_util/extract_lowscore_decoys.py'

#MOD_VOCAB.mod2index acid_mod atom_mod

class Data_Prepare:
    def __init__(self,excel_dir):
        self.excel_dir=excel_dir

        self.json_dir=excel_dir[:-5]+'.json'
        self.secondary_structure = True
        self.chunk_size=30
        self.json = True

        if self.secondary_structure == False:
            self.json_dir=excel_dir[:-5]+'no2.json'


    def get_atommod(self,item):
        smode=None
        amode=None
        atom_mask=[[0 for _ in range(VOCAB.MAX_ATOM_NUMBER)] for _ in range(50)]
        if item['sense mod']!='none' and item['sense mod'] !=0:
            smode=item['sense mod'].split('* ')
            spose=str(item['sense pos']).replace(' ','').split('*')
        if item['anti mod']!='none' and item['anti mod'] !=0:    
            amode=item['anti mod'].split('* ')
            apose=str(item['anti pos']).replace(' ','').split('*')
    
        offset = 25
        #print(item['ID'])

        

        for pos in range(0,len(item['sense seq'])):
            #add sugar
            atom_mask[pos][0]=MOD_VOCAB.mod2index['Standard sugar']
            #add phosphate
            atom_mask[pos][1]=MOD_VOCAB.mod2index['Standard phosphate']
            #add base
            if item['sense raw seq'][pos]=='a':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard adenine']
            if item['sense raw seq'][pos]=='c':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard cytosine']
            if item['sense raw seq'][pos]=='g':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard guanine']
            if item['sense raw seq'][pos]=='u':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard uracil']
            if item['sense raw seq'][pos]=='t':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard thymine']
                atom_mask[pos][0]=MOD_VOCAB.mod2index['2-Deoxyribonucleotide']

        if smode!=None:                
            for i in range(len(smode)):
                #add mod base
                if smode[i]=='Inverted abasic':
                    for pos in str(spose[i]).split(','): 
                        atom_mask[int(pos)-1][2]=0
                        atom_mask[int(pos)-1][2]=0

                if smode[i] in MOD_VOCAB.base_mod:
                    for pos in str(spose[i]).split(','): 
                        if item['sense raw seq'][int(pos)-1]=='a':
                            atom_mask[int(pos)-1][2]=MOD_VOCAB.mod2index[smode[i]]
                        if item['sense raw seq'][int(pos)-1]=='c':
                            atom_mask[int(pos)-1][2]=MOD_VOCAB.mod2index[smode[i]]
                        if item['sense raw seq'][int(pos)-1]=='g':
                            atom_mask[int(pos)-1][2]=MOD_VOCAB.mod2index[smode[i]]
                        if item['sense raw seq'][int(pos)-1]=='u':
                            atom_mask[int(pos)-1][2]=MOD_VOCAB.mod2index[smode[i]]
                        if item['sense raw seq'][int(pos)-1]=='t':
                            atom_mask[int(pos)-1][2]=MOD_VOCAB.mod2index[smode[i]]

                #add mod phosphate
                if smode[i] in MOD_VOCAB.phosphate_mod:
                    for pos in str(spose[i]).split(','): 
                        atom_mask[int(pos)-1][1]=MOD_VOCAB.mod2index[smode[i]]

                #add mod sugar
                if smode[i] in MOD_VOCAB.sugar_mod:
                    for pos in str(spose[i]).split(','): 
                        atom_mask[int(pos)-1][0]=MOD_VOCAB.mod2index[smode[i]]


        _mask=[[0 for _ in range(VOCAB.MAX_ATOM_NUMBER)] for _ in range(25)]
        for pos in range(len(item['anti seq'])):
            #add sugar
            _mask[int(pos)][0]=MOD_VOCAB.mod2index['Standard sugar']
            #add phosphate
            _mask[int(pos)][1]=MOD_VOCAB.mod2index['Standard phosphate']
            #add base
            if item['anti raw seq'][pos]=='a':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard adenine']
            if item['anti raw seq'][pos]=='c':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard cytosine']
            if item['anti raw seq'][pos]=='g':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard guanine']
            if item['anti raw seq'][pos]=='u':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard uracil']
            if item['anti raw seq'][pos]=='t':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard thymine']   
                _mask[int(pos)][0]=MOD_VOCAB.mod2index['2-Deoxyribonucleotide']         
                        


        if amode!=None:
                
            for i in range(len(amode)):
            
               
                if amode[i]=='Inverted abasic': #ch
                    for pos in str(apose[i]).split(','): 
                        _mask[int(pos)-1][2]=0
                        _mask[int(pos)-1][2]=0
                #add mod base
                if amode[i] in MOD_VOCAB.base_mod:
                    for pos in str(apose[i]).split(','): 
                        if item['anti raw seq'][int(pos)-1]=='a':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]
                        if item['anti raw seq'][int(pos)-1]=='c':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]
                        if item['anti raw seq'][int(pos)-1]=='g':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]
                        if item['anti raw seq'][int(pos)-1]=='u':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]
                        if item['anti raw seq'][int(pos)-1]=='t':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]

                #add mod phosphate
                if amode[i] in MOD_VOCAB.phosphate_mod:
                    for pos in str(apose[i]).split(','): 
                        _mask[int(pos)-1][1]=MOD_VOCAB.mod2index[amode[i]]

                #add mod sugar
                if amode[i] in MOD_VOCAB.sugar_mod:
                    for pos in str(apose[i]).split(','): 
                        try:
                            _mask[int(pos)-1][0]=MOD_VOCAB.mod2index[amode[i]]
                        except:
                            print(print(item['ID']),'wrong')
                            return None


        atom_mask[offset:]=_mask

        return atom_mask

    def seq_pre(self,seq):
        return seq.replace('t','u')

    def raw_pre(self,seq):
        return seq.lower().replace(' + ','').replace('d','').replace('y','g').replace('x','t').replace(' ','')

    def get_rpos(self,item):
        pos = [0]
        for i in range(1,1+len(item['sense seq'])):
            pos.append(i)
        for i in range(30,30+len(item['anti seq'])):
            pos.append(i)
        return pos

    def chunk_dataframe(self,df, chunk_size):
        chunks = []
        chunk_count = len(df) // chunk_size
        for i in range(chunk_count):
            chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
        if len(df) % chunk_size != 0:
            chunks.append(df[chunk_count * chunk_size:])
        return chunks


    def process(self):
        df=pd.read_excel(self.excel_dir) 
        df['sense raw seq']=df['sense raw seq'].apply(self.raw_pre)
        df['anti raw seq']=df['anti raw seq'].apply(self.raw_pre)

        df['sense seq']=df['sense raw seq'].apply(self.seq_pre)
        df['anti seq']=df['anti raw seq'].apply(self.seq_pre)


        if self.json==True:

            dfj=df[['ID','sense seq','sense raw seq','anti seq','anti raw seq','PCT','sense mod','sense pos','anti mod','anti pos']]


            dfj['atom_mask']=dfj.apply(self.get_atommod,axis=1) 



            
            dfj=dfj.dropna()
            dfj.to_json(self.json_dir, orient='records', lines=True)

    


   

def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('-f','--filenames', nargs='+', help='train/valid/test set')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    for filename in args.filenames:
        Data_Prepare(filename).process()
  

