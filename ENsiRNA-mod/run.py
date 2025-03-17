# a pipeline for modified siRNA prediction

import argparse
import os
import sys
import torch
import pandas as pd

#input ID
def input_data():
    print("Input ID:\n For example: Givosiran")
    ID = input()

    #input anti-sense sequence
    print("Input anti-sense sequence:\n For example: UAAGAUGAGACACUCUUUCUGGU")
    anti_seq = input()

    anti_mod = {}
    #Input anti-sense modification

    print("Input a type of anti-sense modification and position split by ':' once a time, and input '0' to end")
    print("For example: ")
    print("     {Input '2-Fluoro:2,3,4,6,8,10,12,14,16,18,20'")
    print("     Input Enter to continue")
    print("     Input '2-O-Methyl:1,5,7,9,11,13,15,17,19,21,22,23'")
    print("     Input Enter to continue")
    print("     Input 'Phosphorothioate:2,3,22,23'")
    print("     Input Enter to continue")
    print("     Input '0' to end}")
    while True:
        print("Input anti-sense modification:")
        mod = input()
        if mod == "0":
            break
        mod_type = mod.split(":")[0]
        mod_pos = mod.split(":")[1]
        anti_mod[mod_type] = mod_pos   

    anti_pos = '* '.join([i for i in anti_mod.values()])
    anti_mod = ' * '.join([i for i in anti_mod.keys()])
    #Input sense sequence
    print("Input sense sequence:\n For example: CAGAAAGAGUGUCUCAUCUUA")
    sense_seq = input()

    sense_mod = {}
    #Input sense modification
    print("Input a type of anti-sense modification and position split by ':' once a time, and input '0' to end")
    print("For example: ")
    print("     {Input '2-Fluoro:2,3,4,6,8,10,12,14,16,18,20'")
    print("     Input Enter to continue")
    print("     Input '2-O-Methyl:1,2,3,4,5,6,8,10,12,14,16,17,18,19,20,21'")
    print("     Input Enter to continue")
    print("     Input 'Phosphorothioate:2,3'")
    print("     Input Enter to continue")
    print("     Input '0' to end}")

    while True:
        print("Input sense modification:")
        mod = input()
        if mod == "0":
            break
        mod_type = mod.split(":")[0]
        mod_pos = mod.split(":")[1]
        sense_mod[mod_type] = mod_pos
    pos_sense = '* '.join([i for i in sense_mod.values()])
    mod_sense = ' * '.join([i for i in sense_mod.keys()])

    columns=['ID','source','cc','sense raw seq','sense mod','sense pos','anti raw seq','anti mod','anti pos','PCT','anti length','sense length','cc_norm','group']

    #写入为dataframe
    df = pd.DataFrame(columns=columns)

    #写入到dataframe
    df.loc[0] = [ID,0,0,sense_seq,mod_sense,pos_sense,anti_seq,anti_mod,anti_pos,0,len(anti_seq),len(sense_seq),0,0]
    df.to_excel(f'result/{ID}.xlsx',index=False)

    print(f'Finish data preparation to {ID}.xlsx')
    print(f'Start to generate pdb data')
    return ID

if __name__ == '__main__':
    ID = input_data()
    from data.get_pdb import Data_Prepare
    import os
    now_path = os.getcwd()
    os.makedirs(f'{now_path}/result/pdb_data',exist_ok=True)
    data_prepare = Data_Prepare(f'{now_path}/result/{ID}.xlsx',f'{now_path}/result/pdb_data')
    data_prepare.process()
    print(f'Finish pdb data generation')

    print(f'Start to predict')
    #调用bash脚本
    os.system(f'bash test.sh pkl/checkpoint_1.ckpt result/{ID}.json {now_path}/result {ID}')
    print(f'Finish prediction')


