import torch
from utils.singleton import singleton
#from unimol_tools import UniMolRepr
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem

mod_smile={ 'Standard sugar':'OCC1OCC(O)C1O', #sugar
            'Standard phosphate':'OP(O)(=O)O',
            'Standard thymine':'CC1=CNC(=O)NC1=O',
            'Standard uracil':'C1=CNC(=O)NC1=O',
            'Standard adenine':'C1=NC2=NC=NC(=C2N1)N',
            'Standard cytosine':'C1=C(NC(=O)N=C1)N',
            'Standard guanine':'C1=NC2=C(N1)C(=O)NC(=N2)N',
            
            'Thymidine':'CC1=CNC(=O)NC1=O',            
            'Locked nucleic acid':'OCC12COC(CO1)C2O', #sugar
            'Unlocked nucleic acid':'OCCOC(CO)CO', #sugar
            'unlocked nucleic acid':'OCCOC(CO)CO', #sugar
            '2-O-Methyl':'COC1COC(CO)C1O', #sugar
            '2-O-Methyl ribose':'COC1COC(CO)C1O',
            '2-O-Methylribose':'COC1COC(CO)C1O',
            '2-Fluoro':'OCC1OCC(F)C1O', #sugar
            '2-Deoxyribonucleotide':'OCC1OCCC1O', #sugar

            
            ### ALL
            'Altritol nucleic acid':'OCC12COC(CO1)C2O', #su

            #'1-Aminohexane':'CCCCCCN', # sugar
            #'12-Hydroxydodecyl 2,3,4,6-tetra-O-methyl-b-D-glucopyranoside':'COC1OC(OCCCCCCCCCCCCO)C(OC)C(OC)C1OC', #sugar x
            '2,3-Di-Chlorobenzene':'C1=C(Cl)C(Cl)=CC=C1', #base
            '2,4-bridged nucleic acid':'OCC12COC(CO1)C2O', #sugar
            '2,4-Carbocyclic-Ethylene-bridged nucleic acid-Locked nucleic acid':'OCC12CCCC(CO1)C2O', #sugar
            '2,4-Carbocyclic-Locked nucleic acid-Locked nucleic acid':'OCC12CCC(CO1)C2O',
            '2,4-Difluorobenzene':'C1=C(F)C=C(F)C=C1', #base
            '2,4-Difluorotoluene':'CC1=CC(C)=C(F)C=C1F', #base
            #'2,4-Difluorotoluene-2-Deoxyribonucleotide'  #cat
            '2-Aminoethoxymethyl':'NCCOCOC1COC(CO)C1O', #sugar 11
            '2-Aminoethyl':'NCCOC1COC(CO)C1O', #sugar 1
            '2-Aminopropoxymethyl':'NCCCOCOC1COC(CO)C1O', #sugar 1
            '2-Aminopropyl_sugar':'NCCCOC1C(O)C(CO)OC1', #su
            '2-Aminopropyl_base':'N1C=NC2=C1C=C(F)C=C2F', #base
            '2-Aminopurineribonucleotide':'NC1=NC=C2N=CNC2=N1', #base
            '2-Cyanoethyl':'OCC1OCC(OCC#N)C1O', #sugar
            '2-Deoxy':'OCC1OCCC1O', #sugar
            'DeoxyThymidine':'OCC1OCCC1O', #sugar
            #'2-Deoxy-2-Aminopurine-2-Deoxyribonucleotide', #cat
            '2-Deoxy-2-Fluoro':'OCC1OCC(F)C1O', #sugar
            '2-Deoxy-2-fluororibose':'OCC1OCC(F)C1O', #sugar
            '2-Deoxy-2-Fluoro-4-Thioarabinonucleic acid':'OCC1SCC(F)C1O',#sugar
            '2-Deoxy-2-Fluoroarabinonucleic acid':'OCC1OCC(F)C1O', #sugar
            '2-Deoxy-2-fluorouridine':'OCC1OCC(F)C1O', #sugar
            
            '2-Deoxy-2-N,4-C-Ethylene-Locked nucleic acid':'OCC12CCNC(CO1)C2O',
            '2-Deoxyinosine':'OCC1OCCC1O',   #N1C=NC2=C1NC=NC2=O' #2 Deoxy + inosine !!!!!!!!!!!
            '2-Deoxynebularine':'OCC1OCCC1O', #'N1C=NC2=CN=CN=C12' #2 Deoxy +  !!!!!!!!!!!
            '2-Deoxythymidine':'OCC1OCCC1O', #2 Deoxy +thymidine
            '2-Guanidinoethyl':'NC(N)=NCCC1COC(CO)C1O',
            '2-Hydroxy':'OCC1OCC(O)C1O',
            #'2-Hydroxyethyl 2,3,4,6-tetra-O-Methyl-b-D-Glucopyranoside':'COC1OC(OCCOP(O)([O-])=O)C(OC)C(OC)C1OC', #ph
            '2-Methoxy':'COC1COC(CO)C1O', #sugar ==next
            '2-Methyl':'CC1COC(CO)C1O',
            '2-Methoxyribose':'COC1COC(CO)C1O',
            '2-N-Adamant-1-yl-Methylcarbonyl-2-Amino-Locked Nucleic Acid':'	OCC12CN(C(CO1)C2O)C(=O)CC12CC3CC(CC(C3)C1)C2', #su
            '2-N-Adamant-1-yl-Methylcarbonyl-2-Amino-Locked nucleic acid':'	OCC12CN(C(CO1)C2O)C(=O)CC12CC3CC(CC(C3)C1)C2', #su
            '2-N-Pyren-1-yl Methyl-2-Amino-Locked nucleic acid':'OCC12CN(CC3=CC=C4C=CC5=CC=CC6=CC=C3C4=C56)C(CO1)C2O',
            '2-O-(2-Methoxyethyl)':'COCCOC1COC(CO)C1O', 
            '2-O-(2-methoxyethyl)ribose':'COCCOC1COC(CO)C1O',
            '2-O-Allyl-ribose':'OCC1OCC(OCC=C)C1O', #su
            '2-O-Benzyl':'OCC1OCC(OC2=CC=CC=C2)C1O',
            '2-O-Fluoro':'OCC1OCC(OF)C1O',
            '2-O-Guanidinoethyl-ribose':'NC(N)=NCCOC1COC(CO)C1O',
            '2-O-Guanidinopropyl':'NC(=N)NCCCOC1COC(CO)C1O',
            '2-O-Lysylaminohexyl':'CCC1OCC(OCCCCCCNC(=O)C(CCCCNC(=O)C(F)(F)F)NC(=O)C(F)(F)F)C1O',
            '2-O-methoxyethylribose':'COCCOC1COC(CO)C1O',
            '2-O-Methyl-4-Thioribose':'COC1CSC(CO)C1O',
            '2-Thiouridine':'N1C=CC(=O)NC1=S', #base
            '3-Amino':'NC1C(O)COC1CO',
            '3-O-methylribose':'COC1C(O)COC1CO',
            '4-C- Aminomethyl-2-O-Methylribose':'COC1COC(CN)(CO)C1O',
            '4-C-Hydroxymethyl Deoxyribonucleic acid':'OCC1(CO)OCCC1',
            '4-Methylbenzimidazole':'CC1=CC=CC2=C1N=CN2', #base
            '4-Thioribose':'OCC1SCC(O)C1O', #sugar
            '5-Amino':'NCCCP(O)(=O)OCC1OCC(O)C1O',
            '5-Bromo-Uridine':'N1C=C(Br)C(=O)NC1', #base
            #'5-Cholesterol':'CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C', #5'
            '5-Fluoro-2-Deoxyuridine':'N1C=C(F)C(=O)NC1=O', #base
            '5-Iodo-Uridine':'N1C=C(I)C(=O)NC1=O', #base
            '5-Methylcytosine':'CC1=CNC(=O)N=C1N', #base
            '5-Nitroindole-2-Deoxyribonucleotide':'N1C=CC2=CC(=CC=C12)[N+]([O-])=O', #base
            '5-Nitroindoleribonucleotide':'N1C=CC2=CC(=CC=C12)[N+]([O-])=O', #base
            '5-O-Methyl':'COCC1OCC(O)C1O', #sugar
            '5-O-methylthymidine':'N1C=C(C)C(=O)NC1=O', #base
            '5-Phosphate':'OP(O)(=O)O',  #standard
            '5-phosphate ribose':'OP(O)(=O)O',  #standard
            '5-Thio':'OC1COC(CS)C1O', #sugar
            #'Abasic(2-hydroxymethyl-tetrahydrofuran-3-ol)', #无碱基
            'Alfa-L-Locked nucleic acid':'OCC1OCCC(O)C1O', #su
            'Aminoisonucleotide-Adenine':'NC1OC(CO)C(O)C1', #su
            'Aminoisonucleotide-thymidine':'NC1OC(CO)C(O)C1', #su
            'Anthracene':'C1=CC2=CC3=CC=CC=C3C=C2C=C1', #base
            #'Biotin'
            'Boranophosphate':'BP(O)(=O)O', #ph
            #'Boron cluster':'[B]1234[B]567[B]118[B]229[B]11%10[B]22%11[B]%12%13%14[B]35([B]6%123[B]12%13[C]78%103)[C]49%11%14', #base
            'Cyclohexenyl nucleic acid':'OCC1C=CCCC1O', #sugar
            #'D-Isonucleoside-adenine' #x base in 2'
            #'D-Isonucleoside-uracil'
            'Deoxyadenine':'OCC1OCCC1O', #2Deoxy 
            'Deoxyuridine':'OCC1OCCC1O', #2Deoxy
            'Diaminopurine':'NC1=NC2=C(NC=N2)C(N)=N1', #base
            'Difluorotoluyl nucleotide':'CC1=CC=C(F)C=C1F', #base
            'Dihydrouridine':'O=C1CCNC(=O)N1', #su
            'Dimethoxy-Nitrophenyl Ethyl group':'COC1=C(OC)C=C(C(C)=C1)[N+]([O-])=O', #su
            'Dodecyl derivative':'CCCCCCCCCCCCOP(O)(O)=O', #ph
            'Fluorescein':'OC1=CC2=C(C=C1)C1(OC(=O)C3=CC=CC=C13)C1=C(O2)C=C(O)C=C1', #su
            #'Fluorobenzene'
            'Fluorouridine':'N1C=C(F)C(=O)NC1=O', #base
            'Galactose':'OCC1OC(O)C(O)C(O)C1O', #su
            'Glucosamine analogue':'NC1C(O)C(O)C(CO)OC1OCCOP(O)([O-])=O',#su
            'Hexitol nucleic acid':'OCC1OCCCC1O', #sugar
            #'Hydroxyphenyl-Conjugated' #x
            'Hypoxanthine':'O=C1N=CNC2=C1NC=N2', #base
            'Inosine':'N1C=NC2=C1NC=NC2=O', #base
           # 'Inverted deoxy abasic' #2脱氧无碱基
           # 'Inverted DeoxyThymidine'  #2脱氧
           # 'Inverted thymidine'
            #'L-isonucleoside thymidine'
            #'Lauric acid'
            'Methyl':'CC1C(O)COC1CO', #	3-Methyl  sugar
            'Methylcytosine':'CC1=C(N)NC(=O)N=C1',#base
            'Methyleneamide':'NC(=O)CC1C(O)COC1CO', #su
            'Methyluridine':'CC1=CNC(=O)NC1=O',#base
            #'Morpholinouridine'
            'N-3 Methyluridine':'CN1C(=O)C=CNC1=O', #base
            #'N-hexylhexadecanamide'
            #'N4-Ethyl-N4 2-Deoxy-5-Methylcytidine'
            #'N6-Ethyl-N6 2-Deoxyadenosine'
            'Naphthalene modification':'C1=CC2=CC=CC=C2C=C1',#base
            'Nebularine':'N1C=NC2=CN=CN=C12',#base
            'Oxetane-Locked nucleic acid':'OCC1OC2OC2C1O',#su
            #'Palmitic acid'
            #'Peptide nucleic acids'
            #'Phenyl'
            'Phosphate':'OCC1OCC(O)C1OP(O)([O-])=O', #sugar 3→5 1'
            'Phosphodiester':'OP(O)(=O)O', #stand phi
            'Phosphorothioate':'OP(O)(S)=O',# Phosphate
            'Propynyluridine':'CC#CC1=CNC(=O)NC1=O',#Base
            'Pseudouridine':'C1=CNC(=O)NC1=O', #base
            #'Puromycin'
            #'Pyrene modification'
            'Serinol nucleic acid':'CC(=O)NC(CO)CO', #su
            #'Thymidine analogue'
            #'Triazole-linked nucleic acid'
            'Tricyclodeoxyribonucleic acid':'OC1CCC2(O)CCOC12', #su
            #'Trifluoromethylbenzene'
            'UnLocked nucleic acid':'OCCOC(CO)CO', #sugar
            'Xylo-3-fluororibose':'OCC1OCC(O)C1F',#su
            'Xylo-O-methylribose':'COC1CCOC1CO', #su
            #'[1,1-Biphenyl]-3,5-dimethanol'
            #'[3-(hydroxymethyl)-5-naphthalen-1-ylphenyl]methanol'
            #'[3-(hydroxymethyl)-5-phenanthren-9-ylphenyl]methanol'
            #'[3-(hydroxymethyl)-5-pyren-1-ylphenyl]methanol':
            #'[3-(hydroxymethyl)phenyl]methanol':
            }


sugar_mod=[
        'Hexitol nucleic acid','Locked nucleic acid','Unlocked nucleic acid','2-O-Methyl','2-Fluoro','2-Deoxy','2-Deoxyribonucleotide','4-Thioribose',
        '2,4-bridged nucleic acid','2,4-Carbocyclic-Ethylene-bridged nucleic acid-Locked nucleic acid','2,4-Carbocyclic-Locked nucleic acid-Locked nucleic acid',
        '2-Aminoethoxymethyl','2-Aminoethyl','2-Aminopropoxymethyl','2-Cyanoethyl','2-Deoxy','2-Deoxy-2-Fluoro','2-Deoxy-2-Fluoro-4-Thioarabinonucleic acid',
        '2-Deoxy-2-Fluoroarabinonucleic acid','2-Deoxy-2-fluorouridine','2-Deoxy-2-N,4-C-Ethylene-Locked nucleic acid','2-Guanidinoethyl','2-O-Guanidinopropyl',
        '2-O-Lysylaminohexyl','2-O-methoxyethylribose','2-O-Methyl-4-Thioribose','3-Amino','3-O-methylribose','4-C- Aminomethyl-2-O-Methylribose','4-C-Hydroxymethyl Deoxyribonucleic acid',
        '4-Thioribose','5-Amino','5-O-Methyl','5-Thio','Alfa-L-Locked nucleic acid','Aminoisonucleotide-Adenine','Aminoisonucleotide-thymidine','Cyclohexenyl nucleic acid',
        'Dihydrouridine','Dimethoxy-Nitrophenyl Ethyl group','Fluorescein','Galactose','Glucosamine analogue','Methyl','Methyleneamide','Oxetane-Locked nucleic acid'
        'Serinol nucleic acid','Tricyclodeoxyribonucleic acid','UnLocked nucleic acid','Xylo-3-fluororibose','Xylo-O-methylribose','2-Aminopropyl_sugar',
        '2-Deoxyinosine','2-Deoxynebularine','2-Deoxythymidine','Deoxyadenine','Deoxyuridine','Altritol nucleic acid','2-N-Adamant-1-yl-Methylcarbonyl-2-Amino-Locked Nucleic Acid',
        '2-N-Pyren-1-yl Methyl-2-Amino-Locked nucleic acid','2-O-(2-Methoxyethyl)','2-O-Allyl-ribose','2-O-Benzyl','2-O-Fluoro','2-O-Guanidinoethyl-ribose',
        '2-O-Methyl ribose','2-O-Methylribose','2-Deoxy-2-fluororibose','2-O-(2-methoxyethyl)ribose','unlocked nucleic acid','DeoxyThymidine','2-Methoxyribose',
        '2-N-Adamant-1-yl-Methylcarbonyl-2-Amino-Locked nucleic acid','Phosphate'
        ]

phosphate_mod=['Boranophosphate','Dodecyl derivative','5-Phosphate','5-phosphate ribose','Phosphodiester','Phosphorothioate']

base_mod=[
        '2,3-Di-Chlorobenzene','2,4-Difluorobenzene','2,4-Difluorotoluene','2-Aminopurineribonucleotide','2-Thiouridine','4-Methylbenzimidazole',
        '5-Bromo-Uridine','5-Fluoro-2-Deoxyuridine','5-Iodo-Uridine','5-Methylcytosine','5-Nitroindole-2-Deoxyribonucleotide','5-Nitroindoleribonucleotide',
        '5-O-methylthymidine','Anthracene','Diaminopurine','Difluorotoluyl nucleotide','Fluorouridine','Hypoxanthine','Inosine','Methylcytosine',
        'Methyluridine','N-3 Methyluridine','Naphthalene modification','Nebularine','Pseudouridine','2-Aminopropyl_base','Thymidine','Propynyluridine'
        ]




class MOD:
    def __init__(self,mod_smile,sugar_mod,phosphate_mod,base_mod):
        self.mod_smile = mod_smile
        self.sugar_mod = sugar_mod
        self.phosphate_mod = phosphate_mod
        self.base_mod = base_mod

        self.mod2index,self.index2smile = {},{}

        for i, mod in enumerate(mod_smile.keys()):
            self.mod2index[mod] = i+1

        for i, smile in enumerate(mod_smile.values()):
            self.index2smile[i+1] = smile
        

    def mod_to_index(self,mod):
        return self.mod2index.get(mod, None)

    def index_to_smile(self,index):
        return self.mod2smile.get(index, None)

MOD_VOCAB=MOD(mod_smile,sugar_mod,phosphate_mod,base_mod)

#from transformers import AutoTokenizer, RobertaModel
#from transformers import logging as transformers_logging
#transformers_logging.set_verbosity_error()



class Chembert(MOD):
    def __init__(self,mod_smile,sugar_mod,phosphate_mod,base_mod,device='cuda'):
        super().__init__(mod_smile,sugar_mod,phosphate_mod,base_mod)
        self.embeddings = {}
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("/public2022/tanwenchong/DDBERT/ChemBERTa-77M-MLM",trust_remote_code=True)
        self.chemmodel = RobertaModel.from_pretrained("/public2022/tanwenchong/DDBERT/ChemBERTa-77M-MLM",trust_remote_code=True)
        self.chemmodel = self.chemmodel.to(device)
        self.dim=384

    def mod_to_embedding(self):
        
        with torch.no_grad():
            for key in self.index2smile.keys():
                inputs = self.tokenizer(self.index2smile[key], return_tensors="pt")
                outputs = self.chemmodel(**inputs.to(self.device))
                embedding = outputs.last_hidden_state[0][0]
                
                self.embeddings[key] = embedding
              
        return self.embeddings,self.dim


#Chembert_VOCAB=Chembert(mod_smile,sugar_mod,phosphate_mod,base_mod).mod_to_embedding()

class ChemUnimol(MOD):
    def __init__(self,mod_smile,sugar_mod,phosphate_mod,base_mod,device='cuda'):
        super().__init__(mod_smile,sugar_mod,phosphate_mod,base_mod)
        self.transformer_emb = {}
        self.rdkit_emb = {}
        self.device = device
        self.clf = UniMolRepr(data_type='molecule', remove_hs=False)
        self.unimol_dim = 512

    
    def mod_to_embedding(self):
        smiles_list = list(self.index2smile.values())
        unimol_repr = self.clf.get_repr(smiles_list, return_atomic_reprs=True)
        for i, key in enumerate(self.index2smile.keys()):
            self.transformer_emb[key] = unimol_repr['cls_repr'][i]
              
        return self.transformer_emb,self.unimol_dim

class ChemRdkit(MOD):
    def __init__(self,mod_smile,sugar_mod,phosphate_mod,base_mod,device='cuda'):
        super().__init__(mod_smile,sugar_mod,phosphate_mod,base_mod)
        self.rdkit_emb = {}
        self.device = device
        self.fp = 'Morgan'
        self.dim = 512


    def mol_to_mogan(self):
        mods=[Chem.MolFromSmiles(mod) for mod in self.index2smile.values()]

        
            
        for i, key in enumerate(self.index2smile.keys()):
            #self.rdkit_emb[key] = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mods[i], 2, nBits=self.dim)).to(self.device)
            self.rdkit_emb[key] = AllChem.GetMorganFingerprintAsBitVect(mods[i], 2, nBits=self.dim)
                                                
            
        return self.rdkit_emb , self.dim

    def mol_to_maccs(self):
        mods=[Chem.MolFromSmiles(mod) for mod in self.index2smile.values()]
            
        for i, key in enumerate(self.index2smile.keys()):
            self.rdkit_emb[key] = torch.tensor(MACCSkeys.GenMACCSKeys(mods[i])).to(self.device)
                                           

        self.dim = 167
           

        return self.rdkit_emb , self.dim



    def mol_to_discriptor(self,mol):
 
        mw = Descriptors.MolWt(mol)  # Molecular Weight
        xlogp = Descriptors.MolLogP(mol)  # XLogP3
        hbd = Descriptors.NumHDonors(mol)  # Hydrogen Bond Donor Count
        hba = Descriptors.NumHAcceptors(mol)  # Hydrogen Bond Acceptor Count
        emass = Descriptors.ExactMolWt(mol)  # Exact Mass
        tpsa = Descriptors.TPSA(mol)  # Topological Polar Surface Area
        hac = Descriptors.HeavyAtomCount(mol)  # Heavy Atom Count



        feature = torch.tensor([mw, xlogp, hbd, hba, emass,tpsa, hac], 
                                dtype=torch.float).to(self.device)
    
        return feature

    def mod_to_feature(self):

        dim = 7
        mods=[Chem.MolFromSmiles(mod) for mod in self.index2smile.values()]
        for i, key in enumerate(self.index2smile.keys()):
            self.rdkit_emb[key] = self.mol_to_discriptor(mods[i])
        
        return self.rdkit_emb,dim
            





#ChemUnimol_VOCAB=ChemUnimol(mod_smile,sugar_mod,phosphate_mod,base_mod).mod_to_embedding()

MOGANRdkit_VOCAB=ChemRdkit(mod_smile,sugar_mod,phosphate_mod,base_mod).mol_to_mogan()
#MACCSRdkit_VOCAB=ChemRdkit(mod_smile,sugar_mod,phosphate_mod,base_mod).mol_to_maccs()

#RdkitDis_VOCAB=ChemRdkit(mod_smile,sugar_mod,phosphate_mod,base_mod).mod_to_feature()
