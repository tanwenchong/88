import argparse
import pandas as pd
from Bio.Seq import Seq
import re
#python get_siRNA.py -i /public2022/tanwenchong/rna/EnSIRNA-main/data/mRNA.fasta -o 
parser=argparse.ArgumentParser()
parser.add_argument('-i','--input',type=str,required=True,help='input mrna fasta file')
parser.add_argument('-s','--sirna', type=str, nargs='+', default=None)
parser.add_argument('-o','--output',type=str,required=True,help='output file')
args=parser.parse_args()

mrnas={}
with open(args.input,'r') as f:
    for line in f:
        if line.startswith('>'):
            mrna_id=line.strip().split('>')[1]
            mrnas[mrna_id]=''
        else:
            mrnas[mrna_id]+=line.strip()

#make 19mer siRNA
columns=['siRNA','anti seq','sense seq','mRNA_seq','position','efficacy']
siRNA=[]
if args.sirna is None:
    for mrna_id,mrna in mrnas.items():
        mrna=mrna.replace('T','U')
        for i in range(len(mrna)-18):
                sense=mrna[i:i+19] #str
                #超过6个重复
                if len(re.findall(r'(.)\1{6,}', sense))>0:
                    continue
                anti = str(Seq(sense).reverse_complement())
                siRNA_id=mrna_id+'_'+str(i)
                siRNA.append([siRNA_id,anti,sense,mrna,i,0])
else:
    for mrna_id,mrna in mrnas.items():
        for sirna in args.sirna:
            mrna=mrna.replace('T','U')
            sense=sirna.upper().replace('T','U')[:19]
            anti = str(Seq(sense).reverse_complement())
            position=mrna.find(sense)
            siRNA_id=mrna_id+'_'+str(position)
            siRNA.append([siRNA_id,anti,sense,mrna,position,0])

df=pd.DataFrame(siRNA,columns=columns)
df.to_csv(f'{args.output}/{args.input.split("/")[-1].split(".")[0]}.csv',index=False)
csv_path=f'{args.output}/{args.input.split("/")[-1].split(".")[0]}.csv'
#输出给bash
print(csv_path)


