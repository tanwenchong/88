import pandas as pd

import json
from utils.rna_utils import VOCAB
import argparse
import os

import torch
import RNA
import re
import subprocess
import multiprocessing


#example （5'->3' and 3'->5'）
#seq1 = "CUUACGCUGAGUACUUCGA".lower()
#seq2 = "GAAUGCGACUCAUGAAGCU".lower()[::-1]

#python -m data.get_pdb 
RF='/app/ENsiRNA-main/rosetta/rosetta.binary.linux.release-371'
FF='/app/ENsiRNA-main/rosetta/rosetta.binary.linux.release-371/main/source/bin/rna_denovo.static.linuxgccrelease'
EX='/app/ENsiRNA-main/rosetta/rosetta.binary.linux.release-371/main/tools/rna_tools/silent_util/extract_lowscore_decoys.py'

#MOD_VOCAB.mod2index acsiRNA_mod atom_mod

class Data_Prepare:
    def __init__(self,excel_dir,pdb_dir):
        self.excel_dir=excel_dir
        self.pdb_dir=pdb_dir

        self.json_dir=excel_dir[:-4]+'.json'
        self.secondary_structure = True
        self.chunk_size=None
        self.num_cores = multiprocessing.cpu_count()
        self.json = True

        if self.secondary_structure == False:
            self.json_dir=excel_dir[:-4]+'no2.json'


    def get_path(self,siRNA):
        return f"{self.pdb_dir}/{siRNA}.pdb"


    def raw_pre(self,seq):
        return seq.lower().replace(' + ','').replace('d','').replace('t','u').replace(' ','')

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

    def get_anti_start(self,data):
        seq1=data['sense seq']
        seq2=data['anti seq']
        com=f" echo -e \"{seq1}\n{seq2}\n\" | RNAplex "
        raw_se=subprocess.run(com,shell=True,capture_output=True, text=True).stdout 
        secondary_seq1 = re.split(r'\s+', raw_se)[0].split('&')[0]
        secondary_seq2 = re.split(r'\s+', raw_se)[0].split('&')[1]
        seq2_ = re.split(r'\s+', raw_se)[3].split(',')
        seq1_ = re.split(r'\s+', raw_se)[1].split(',')
        anti1=0
        anti2=0
        s1=0
        s2=0
        flag=0
        for i in  secondary_seq2:
            if i=='.':
                anti1+=1
                flag=1
            else:
                if flag==1:
                    anti1+=len(seq2[:int(int(seq2_[0])-1)])
                break    
        for i in  secondary_seq2[::-1]:
            if i=='.':
                flag=-1
                anti2-=1
            else:
                if flag==-1:
                    anti2-=len(seq2[int(seq2_[1]):])
                break  
        flag=0
        for i in  secondary_seq1:
            if i=='.':
                flag=1
                s1+=1
            else:
                if flag==1:
                    s1+=len(seq1[:int(seq1_[0])-1])
                break
        for i in  secondary_seq1[::-1]:
            if i=='.':
                flag=-1
                s2-=1
            else:
                if flag==-1:
                    s2-=len(seq1[int(seq1_[1]):])
                break
        seq2_ = re.split(r'\s+', raw_se)[3].split(',')
        sec_pos = [1000]
        padlen = int((61 - len(seq1)) / 2)
        chain = [0]
        for i in range(-padlen,len(seq1)+padlen): #mrna
            sec_pos.append(i)
            chain.append(1)
        sec_pos.append(2000)
        chain.append(2)
        for i in range(len(seq1)): #sense
            sec_pos.append(i)
            chain.append(3)
        for i in range(s2+anti1+len(seq1)-1,s1+anti2-1,-1): #anti
            sec_pos.append(i)
            chain.append(3)
  
        if len(sec_pos) != 61+len(seq2)+len(seq1)+1+1:
            print('!=',data['siRNA'],len(sec_pos),len(seq2))
            return None
        return sec_pos,chain

    def process(self):
        df=pd.read_csv(self.excel_dir) 
        df['sense seq']=df['sense seq'].apply(self.raw_pre)
        df['anti seq']=df['anti seq'].apply(self.raw_pre)


        df[['start', 'chain']] = df.apply(self.get_anti_start, axis=1, result_type='expand')

        total_rows = len(df)
        self.chunk_size = max(1, total_rows // self.num_cores)
        chunks = self.chunk_dataframe(df, self.chunk_size)
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            drops=pool.map(self.get_data, chunks)

        #print(drops)

        #df=df.drop(index=drops)
        if self.json==True:

            dfj=df[['siRNA','mRNA_seq','position','sense seq','anti seq','efficacy','start','chain']] #delect marker

            dfj['pdb_data_path']=dfj['siRNA'].apply(self.get_path)
            dfj['efficacy']=dfj['efficacy'].apply(lambda x: x if x > 0 else 0)
            dfj=dfj.dropna()
            dfj.to_json(self.json_dir, orient='records', lines=True)

    def get_data(self,df):
        
        drops=[]
 
        for index, row in df.iterrows():
            if os.path.exists(f"{self.pdb_dir}/{row['siRNA']}.pdb")==True:
                continue
            if os.path.exists(f"{self.pdb_dir}/{row['siRNA']}")==True:
                continue
            if self.secondary_structure ==True:
                if self.get_secondary_structure(row)==False:
                    print(f"drop{self.pdb_dir}/{row['siRNA']}.pdb")
                    drops.append(index)

            else:
                if self.get_structure(row)==False:
                    print(f"drop{self.pdb_dir}/{row['siRNA']}.pdb")
                    drops.append(index)

        return drops
    



    

    def get_structure(self,data):
    
        seq1=data['sense seq']
        seq2=data['anti seq']

        pose = assembler.build_init_pose(seq1, seq2)
        pose.dump_pdb(f"{self.pdb_dir}/{data['siRNA']}.pdb")
        if os.path.exists(f"{self.pdb_dir}/{data['siRNA']}.pdb"):
            return True
        else:
            return False

    def get_secondary_structure(self,data):
        #if os.path.exists(f"{self.pdb_dir}/{data['siRNA']}"):
        #    return True
    
        seq1=data['sense seq']
        seq2=data['anti seq']
        seq=seq1+' '+seq2

        #sencondary_seq=RNA.duplexfold(seq1,seq2).structure.replace('&',' ')
        com=f" echo -e \"{seq1}\n{seq2}\n\" | RNAplex "
        raw_se=subprocess.run(com,shell=True,capture_output=True, text=True).stdout

        secondary_seq1 = re.split(r'\s+', raw_se)[0].split('&')[0]
        
        secondary_seq2 = re.split(r'\s+', raw_se)[0].split('&')[1]
        
        
        seq1_ = re.split(r'\s+', raw_se)[1].split(',')
        seq2_ = re.split(r'\s+', raw_se)[3].split(',')

        secondary_seq1 = ''.join(['.' for _ in seq1[:int(seq1_[0])-1]]) + secondary_seq1 + ''.join(['.' for _ in seq1[int(seq1_[1]):]])
        secondary_seq2 = ''.join(['.' for _ in seq2[:int(seq2_[0])-1]]) + secondary_seq2 + ''.join(['.' for _ in seq2[int(seq2_[1]):]])

        secondary_seq = secondary_seq1 + ' ' + secondary_seq2

        os.mkdir(f"{self.pdb_dir}/{data['siRNA']}")
        os.chdir(f"{self.pdb_dir}/{data['siRNA']}")

        subprocess.run([FF,'-sequence',seq,'-secstruct',secondary_seq,'-minimize_rna'])
        subprocess.run(['python',EX,'default.out','-rosetta_folder',RF,'1'])
        subprocess.run(['cp','default.out.1.pdb',f"{self.pdb_dir}/{data['siRNA']}.pdb"])
      
        #os.chdir(f"/public2022/tanwenchong/rna/EnModSIRNA-1main")
        subprocess.run(['rm','-r',f"{self.pdb_dir}/{data['siRNA']}"])
        

        if os.path.exists(f"{self.pdb_dir}/{data['siRNA']}.pdb"):
            return True
        else:
            return False

   

def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('-f','--filenames', nargs='+', help='train/valsiRNA/test set')
    parser.add_argument('-p','--pdb_dir', type=str, default=None, help='Path to save processed data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    for filename in args.filenames:
        Data_Prepare(filename,args.pdb_dir).process()
  

