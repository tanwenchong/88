import argparse


# examle:P(mU)#(fA)#(mA)(fA)(mU)(fU)(mU)(fG)(mG)(fA)(mG)(fA)(mU)#(fC)#(mC)#(fG)#(mA)#(fG)#(mA)#(fG)
#可以不需要括号 PmU#fA#mAfAmUfUmUfGmGfAmGfAmU#fC#mC#fG#mA#fG#mA#fG 修饰+核酸，修饰+核酸

class Change:
    def __init__(self):
        self.s2m =  {
                    '#':'Phosphorothioate',
                    'm':'2-O-Methyl',
                    'f':'2-Fluoro',
                    }

        self.rna=['A','C','G','U']

        self.mod2smile={
                        '2-Methyl':'CC1C(O)OC(CO)C1O',
                        '2-O-Methyl':'COC1C(O)OC(CO)C1O',
                        'Phosphorothioate':'OCC1OCC(O)C1OP(O)(S)=O',
                        '2-Fluoro':'OCC1OCC(F)C1O'
        }

    def seq_to_mod(self,inputs):
        seq=''
        pos=1
        Phosphorothioate=[]
        Methyl=[]
        Fluoro=[]
        mod=[]
        mod_pos=[]

        for i in list(inputs):
            if i == '#':
                Phosphorothioate.append(str(pos))
            elif i == 'm':
                Methyl.append(str(pos))
            elif i == 'f':
                Fluoro.append(str(pos))

            if i in self.rna:
                seq+=i
                pos+=1

        if Phosphorothioate!=[]:
            mod.append(self.s2m['#'])
            mod_pos.append(','.join(Phosphorothioate))
        if Methyl!=[]:
            mod.append(self.s2m['m'])
            mod_pos.append(','.join(Methyl))
        if Fluoro!=[]:
            mod.append(self.s2m['f'])
            mod_pos.append(','.join(Fluoro))

        return seq,'* '.join(mod),' * '.join(mod_pos)


    def mod_to_CM(self,sen_mod,anti_mod,sen_pos,anti_pos): 
        sen_smiles=[]
        anti_smiles=[]
        for mod in sen_mod.split('* '):
            sen_smiles.append(self.mod2smile[mod])
        for mod in anti_mod.split('* '):
            anti_smiles.append(self.mod2smile[mod])
        spos=[]
        apos=[]
        for pos in sen_pos.split(' * '):
            spos.append(pos)
        for pos in anti_pos.split(' * '):
            apos.append(pos)
        
        mod=';'.join(sen_smiles)+'*'+';'.join(anti_smiles)
        pos=';'.join(spos)+'*'+';'.join(apos)

        return mod+'__'+pos





        
    




def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('-s','--seq', type=str, required=True, help='siRNA seq')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    seq,mod,pos=Change().seq_to_mod(args.seq)
    print('seq',seq)
    print('mos',mod)
    print('mod_pos',mod_pos)
