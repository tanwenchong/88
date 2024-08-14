from copy import copy, deepcopy
import math
import os
from typing import Dict, List, Tuple
import requests
import torch
import torch.nn.functional as F
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.Chain import Chain as BChain
from Bio.PDB.Residue import Residue as BResidue
from Bio.PDB.Atom import Atom as BAtom




class NucleicAcid:
    def __init__(self, symbol: str, sidechain: List[str], idx=0):
        self.symbol = symbol
        self.idx = idx
        self.sidechain = sidechain

    def __str__(self):
        return f'{self.idx} {self.symbol} {self.sidechain}'

class NucleicAcidVocab:

    MAX_ATOM_NUMBER = 3   # sugae pho base

    def __init__(self):
        self.backbone_atoms = ["C4'","P"]
        self.PAD, self.MASK = '#', '*'
        self.BOS = '+'  # begin of rna
        specials = [self.PAD, self.MASK, self.BOS]
        acid = ['A','C','U','G']

        # max number of sidechain atoms: 10        
        self.atom_pad, self.atom_mask = 'p', 'm'
        self.atom_pos_mask, self.atom_pos_bb, self.atom_pos_pad = 'm', 'b', 'p'
        base_map = {
            'A': ['N9'],
            'C': ['N1'],
            'G': ['N9'],
            'U': ['N1'],
        }
        self.chi_angles_atoms = {
            "A": [],
            "C": [],
            "G": [],
            "U": []
            }
        self.sidechain_bonds = {
            "A": { "C4'": ["N9"]},
            "C": { "C4'": ["N1"]},
            "G": { "C4'": ["N9"]},
            "U": { "C4'": ["N1"]},
        }


        _all = acid + specials
        self.nacids = [NucleicAcid(symbol, base_map.get(symbol, [])) for symbol in _all]
        self.symbol2idx, self.abrv2idx = {}, {}
        for i, aa in enumerate(self.nacids):
            self.symbol2idx[aa.symbol] = i
            aa.idx = i
        self.special_mask = [0 for _ in acid] + [1 for _ in specials]

        # atom level vocab
        self.idx2atom = [self.atom_pad, self.atom_mask, "C","P","O","N"]
        self.idx2atom_pos = [self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_bb, '1','9']
        self.atom2idx, self.atom_pos2idx = {}, {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        for i, atom_pos in enumerate(self.idx2atom_pos):
            self.atom_pos2idx[atom_pos] = i
    


    def symbol_to_idx(self, symbol):
        symbol = symbol.upper()
        return self.symbol2idx.get(symbol, None)
    
    def idx_to_symbol(self, idx):
        return self.nacids[idx].symbol

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_mask_idx(self):
        return self.symbol_to_idx(self.MASK)
    
    def get_special_mask(self):
        return copy(self.special_mask)

    def get_atom_type_mat(self):
        atom_pad = self.get_atom_pad_idx()
        mat = []
        for i, aa in enumerate(self.nacids):
            atoms = [atom_pad for _ in range(self.MAX_ATOM_NUMBER)]
            if aa.symbol == self.PAD:
                pass
            elif self.special_mask[i] == 1:  # specials
                atom_mask = self.get_atom_mask_idx()
                atoms = [atom_mask for _ in range(self.MAX_ATOM_NUMBER)]
            else:
                for aidx, atom in enumerate(self.backbone_atoms + aa.sidechain):
                    atoms[aidx] = self.atom_to_idx(atom[0])
            mat.append(atoms)
        return mat

    def get_atom_pos_mat(self):
        atom_pos_pad = self.get_atom_pos_pad_idx()
        mat = []
        for i, aa in enumerate(self.nacids):
            aps = [atom_pos_pad for _ in range(self.MAX_ATOM_NUMBER)]
            if aa.symbol == self.PAD:
                pass
            elif self.special_mask[i] == 1:
                atom_pos_mask = self.get_atom_pos_mask_idx()
                aps = [atom_pos_mask for _ in range(self.MAX_ATOM_NUMBER)]
            else:
                aidx = 0
                for _ in self.backbone_atoms:
                    aps[aidx] = self.atom_pos_to_idx(self.atom_pos_bb)
                    aidx += 1
                for atom in aa.sidechain:
                    aps[aidx] = self.atom_pos_to_idx(atom[1])
                    aidx += 1
            mat.append(aps)
        return mat

    def get_sidechain_info(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return copy(self.nacids[idx].sidechain)
    
    def get_sidechain_geometry(self, symbol):
        
        chi_angles_atoms = copy(self.chi_angles_atoms[symbol])
        sidechain_bonds = self.sidechain_bonds[symbol]
        return (chi_angles_atoms, sidechain_bonds)
    
    def get_atom_pad_idx(self):
        return self.atom2idx[self.atom_pad]
    
    def get_atom_mask_idx(self):
        return self.atom2idx[self.atom_mask]
    
    def get_atom_pos_pad_idx(self):
        return self.atom_pos2idx[self.atom_pos_pad]

    def get_atom_pos_mask_idx(self):
        return self.atom_pos2idx[self.atom_pos_mask]
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def idx_to_atom_pos(self, idx):
        return self.idx2atom_pos[idx]
    
    def atom_pos_to_idx(self, atom_pos):
        return self.atom_pos2idx[atom_pos]

    def get_num_atom_type(self):
        return len(self.idx2atom)
    
    def get_num_atom_pos(self):
        return len(self.idx2atom_pos)

    def get_num_amino_acid_type(self):
        return len(self.special_mask) - sum(self.special_mask)

    def __len__(self):
        return len(self.symbol2idx)


VOCAB = NucleicAcidVocab()

class Acid:
    def __init__(self, symbol: str, coordinate: Dict, _id: Tuple):
        self.symbol = symbol
        self.coordinate = coordinate
        self.sidechain = VOCAB.get_sidechain_info(symbol)
        self.id = _id  # (residue_number, insert_code)

    def get_symbol(self):
        return self.symbol

    def get_coord(self, atom_name):
        return copy(self.coordinate[atom_name])

    def get_coord_map(self) -> Dict[str, List]:
        return deepcopy(self.coordinate)

    def get_backbone_coord_map(self) -> Dict[str, List]:
        coord = { atom: self.coordinate[atom] for atom in self.coordinate if atom in VOCAB.backbone_atoms }
        return coord

    def get_sidechain_coord_map(self) -> Dict[str, List]:
        coord = {}
        for atom in self.sidechain:
            if atom in self.coordinate:
                coord[atom] = self.coordinate[atom]
        return coord

    def get_atom_names(self):
        return list(self.coordinate.keys())

    def get_id(self):
        return self.id

    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_coord(self, coord):
        self.coordinate = deepcopy(coord)

    def dist_to(self, residue):  # measured by nearest atoms
        xa = np.array(list(self.get_coord_map().values()))
        xb = np.array(list(residue.get_coord_map().values()))
        if len(xa) == 0 or len(xb) == 0:
            return math.nan
        dist = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
        return np.min(dist)

    def to_bio(self):
        _id = (' ', self.id[0], self.id[1])
        residue = BResidue(_id, VOCAB.symbol_to_abrv(self.symbol), '    ')
        atom_map = self.coordinate
        for i, atom in enumerate(atom_map):
            fullname = ' ' + atom
            while len(fullname) < 4:
                fullname += ' '
            bio_atom = BAtom(
                name=atom,
                coord=np.array(atom_map[atom], dtype=np.float32),
                bfactor=0,
                occupancy=1.0,
                altloc=' ',
                fullname=fullname,
                serial_number=i,
                element=atom[0]  # not considering symbols with 2 chars (e.g. FE, MG)
            )
            residue.add(bio_atom)
        return residue

    def __iter__(self):
        return iter([(atom_name, self.coordinate[atom_name]) for atom_name in self.coordinate])


class Chain:
    def __init__(self, _id, residues: List[Acid]):
        self.residues = residues
        self.seq = ''
        self.id = _id
        for residue in residues:
            self.seq += residue.get_symbol()

    def set_id(self, _id):
        self.id = _id

    def get_id(self):
        return self.id

    def get_seq(self):
        return self.seq

    def get_span(self, i, j):  # [i, j)
        i, j = max(i, 0), min(j, len(self.seq))
        if j <= i:
            return None
        else:
            residues = deepcopy(self.residues[i:j])
            return Peptide(self.id, residues)

    def get_residue(self, i):
        return deepcopy(self.residues[i])
    


    def set_residue_coord(self, i, coord):
        self.residues[i].set_coord(coord)

    def set_residue_translation(self, i, vec):
        coord = self.residues[i].get_coord_map()
        for atom in coord:
            ori_vec = coord[atom]
            coord[atom] = [a + b for a, b in zip(ori_vec, vec)]
        self.set_residue_coord(i, coord)

    def set_residue_symbol(self, i, symbol):
        self.residues[i].set_symbol(symbol)
        self.seq = self.seq[:i] + symbol + self.seq[i+1:]

    def set_residue(self, i, symbol, coord):
        self.set_residue_symbol(i, symbol)
        self.set_residue_coord(i, coord)

    def to_bio(self):
        chain = BChain(id=self.id)
        for residue in self.residues:
            chain.add(residue.to_bio())
        return chain

    def __iter__(self):
        return iter(self.residues)

    def __len__(self):
        return len(self.seq)

    def __str__(self):
        return self.seq


class RNA:
    def __init__(self, pdb_id, chains):
        self.pdb_id = pdb_id
        self.chains = chains

    @classmethod
    def from_pdb(cls, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb_path)
        pdb_id = structure.header['idcode'].upper().strip()
        if pdb_id == '':
            # deduce from file name
            pdb_id = os.path.split(pdb_path)[1].split('.')[0] + '(filename)'

        chains = {}
        for chain in structure.get_chains():
            _id = chain.get_id()
            residues = []
            has_non_residue = False
            for residue in chain:
                symbol = residue.get_resname()
                hetero_flag, res_number, insert_code = residue.get_id()
                if hetero_flag != ' ':
                    continue   # residue from glucose or water
            
                # filter Hs because not all data include them
                atoms = { atom.get_id(): atom.get_coord() for atom in residue if atom.element != 'H' }
                residues.append(Acid(
                    symbol, atoms, (res_number, insert_code)
                ))
               
            if has_non_residue or len(residues) == 0:  # not a peptide
                continue
            chains[_id] = Chain(_id, residues)
        return cls(pdb_id, chains)

    def get_id(self):
        return self.pdb_id

    def num_chains(self):
        return len(self.chains)

    def get_chain(self, name):
        if name in self.chains:
            return deepcopy(self.chains[name])
        else:
            return None

    def get_chain_names(self):
        return list(self.chains.keys())

    def to_bio(self):
        structure = BStructure(id=self.pdb_id)
        model = BModel(id=0)
        for name in self.chains:
            model.add(self.chains[name].to_bio())
        structure.add(model)
        return structure

    def to_pdb(self, path, atoms=None):
        if atoms is None:
            bio_structure = self.to_bio()
        else:
            prot = deepcopy(self)
            for _, chain in prot:
                for residue in chain:
                    coordinate = {}
                    for atom in atoms:
                        if atom in residue.coordinate:
                            coordinate[atom] = residue.coordinate[atom]
                    residue.coordinate = coordinate
            bio_structure = prot.to_bio()
        io = PDBIO()
        io.set_structure(bio_structure)
        io.save(path)

    def __iter__(self):
        return iter([(c, self.chains[c]) for c in self.chains])

    def __eq__(self, other):
        if not isinstance(other, Protein):
            raise TypeError('Cannot compare other type to Protein')
        for key in self.chains:
            if key in other.chains and self.chains[key].seq == other.chains[key].seq:
                continue
            else:
                return False
        return True

    def __str__(self):
        res = self.pdb_id + '\n'
        for seq_name in self.chains:
            res += f'\t{seq_name}: {self.chains[seq_name]}\n'
        return res

class RNAFeature:
    def __init__(self):
        self.phychem={
        '+':[0 for i in range(10)],
        't':[322.21,-2.8,4,8,322.05660244,322.05660244,146,21,529,0],
        'c':[323.20,-3.4,5,8,323.05185141,323.05185141,175,21,531,0],
        'g':[363.22,-3.5,6,10,363.05799942,363.05799942,202,24,598,0],
        'a':[347.22,-3.5,5,11,347.06308480,347.06308480,186,23,481,0],
        'u':[363.22,-3.5,6,10,363.05799942,363.05799942,202,24,598,0],
        }

        self.sym2idx={
        '+':0,
        'a':1,
        'c':2,
        'g':3,
        'u':4,
        't':5
        }

    def seq_to_raw(self,sense_seq,anti_seq):
        S = [self.sym2idx['+']]
        for i in sense_seq:
            S.append(self.sym2idx[i])
        for i in anti_seq:
            S.append(self.sym2idx[i])        

        #S = F.one_hot(torch.tensor(S), num_classes=6)

        return S

    def seq_to_phychem(self,sense_seq,anti_seq):

        S = [self.phychem['+']]
        for i in sense_seq:
            S.append(self.phychem[i])
        for i in anti_seq:
            S.append(self.phychem[i])

        S = torch.tensor(S,dtype=torch.float)      

        return S

RNAFeature = RNAFeature()



if __name__ == '__main__':
    import sys
    pdb_path= sys.argv[1]
    rna=RNA.from_pdb(pdb_path)
    print(rna)