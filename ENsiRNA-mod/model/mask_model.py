import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_scatter import scatter_mean

from utils.rna_utils import VOCAB
from utils.nn_utils import SeparatedNucleicAcidFeature,EasyNucleicAcidFeature,CoordNormalizer
from utils.nn_utils import GMEdgeConstructor
from utils.nn_utils import _knn_edges
#from sklearn.metrics import roc_auc_score
import numpy as np
from model.am_egnn3 import AMEGNN


class RNAmaskModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9,
                 n_layers=3,  dropout=0.1,MLM=False) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        atom_embed_size = 16
        
        self.rna_feature = SeparatedNucleicAcidFeature(
            int(embed_size/2), atom_embed_size, #embed_size
            relative_position=True, #True
            fix_atom_weights=False,
            edge_constructor=GMEdgeConstructor,
            feature='MOGAN',
            atommod=False)

        self.normalizer = CoordNormalizer()

        self.num_classes = num_classes
        self.gnn=AMEGNN(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=0.1, dense=True)

        #self.gnn= GAT(embed_size, hidden_size, out_channels=hidden_size, num_heads=4)

        self.pred1=nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.pred2=nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.pred3=nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        
        self.bos_idx=VOCAB.symbol_to_idx(VOCAB.BOS)
        self.MLM=MLM

        if self.MLM==True:
            self.ffn_rna = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )
        self.c = nn.Linear(1, 1)
        

    def forward(self,S,X,rna_pos,sec_pos,lengths,pct,marker,smask,atom_mask,mod_mask,rna_raw,chain_id,mod,cc,FM,mask_ratio=0): #residue_pos → rna_pos
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        H_0, rnamod_embedding,(ctx_edges,att_edges), (atom_embeddings, atom_weights) = self.rna_feature(X, S, batch_id, self.k_neighbors, rna_pos ,atom_mask , rna_raw,chain_id,mod,FM,sec_pos,mod_mask)
        row, col = ctx_edges
        mod_h, x, _ = self.gnn(torch.cat([H_0,rnamod_embedding],dim=-1), X, ctx_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)#torch.cat([H_0,rnamod_embedding],dim=-1)
        h_ = mod_h[smask]  
        logits2=self.pred2(h_).squeeze()
        probs2 = torch.sigmoid(logits2)
        

        if probs2.dim() == 0:
      
            probs2=probs2.unsqueeze(0)

        mod_loss = F.smooth_l1_loss(probs2, pct) #/ smask.sum()
        loss = mod_loss
        return loss


    def test(self,S,X,rna_pos,sec_pos,lengths,pct,marker,smask,atom_mask,mod_mask,rna_raw,chain_id,mod,cc,FM):
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        
        H_0, rnamod_embedding,(ctx_edges,att_edges), (atom_embeddings, atom_weights) = self.rna_feature(X, S, batch_id, self.k_neighbors, rna_pos ,atom_mask , rna_raw,chain_id,mod,FM,sec_pos,mod_mask)

        row, col = ctx_edges

        mod_h, x, mid_h =self.gnn(torch.cat([H_0,rnamod_embedding],dim=-1), X, ctx_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)
        h_ = mod_h[smask] #* cc
        logits2=self.pred2(h_).squeeze()
        probs2 = torch.sigmoid(logits2)

        if probs2.dim() == 0:
            probs2=probs2.unsqueeze(0)

        return probs2,H_0

