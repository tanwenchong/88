import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_scatter import scatter_mean
from utils.rna_utils import VOCAB
from utils.nn_utils import SeparatedNucleicAcidFeature
from utils.nn_utils import GMEdgeConstructor
from utils.nn_utils import _knn_edges
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
            embed_size, atom_embed_size, #embed_size
            relative_position=False,
            fix_atom_weights=False,
            edge_constructor=GMEdgeConstructor,
            feature='MOGAN',
            atommod=False)


        self.num_classes = num_classes
        self.gnn=AMEGNN(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=0.1, dense=True)

        self.pred1=nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.pred2=nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size*2, hidden_size*2),
                nn.SiLU(),
                nn.Linear(hidden_size*2, 1)
            )
        self.pred3=nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        

    def forward(self,S,X,rna_pos,sec_pos,lengths,pct,smask,single_embeddings,duplex_embeddings,chain,energys): #residue_pos → rna_pos
        
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        H_0,(sirna_edges,mrna_edges), (atom_embeddings, atom_weights) = self.rna_feature(X, S, batch_id, self.k_neighbors, rna_pos ,sec_pos,chain,energys,single_embeddings,duplex_embeddings)
        mod_h, x, _ = self.gnn(H_0, X, sirna_edges,mrna_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)#torch.cat([H_0,rnamod_embedding],dim=-1)
        h_ = torch.cat([mod_h[smask][0::2,:],mod_h[smask][1::2,:]],dim=1) 
        logits2=self.pred2(h_).squeeze()
        probs2 = torch.sigmoid(logits2)
        if probs2.dim() == 0:
            probs2=probs2.unsqueeze(0)
        loss = F.smooth_l1_loss(probs2, pct) #/ smask.sum()
        return loss

    def test(self,S,X,rna_pos,sec_pos,lengths,pct,smask,single_embeddings,duplex_embeddings,chain,energys):

        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch 
        H_0,(sirna_edges,mrna_edges), (atom_embeddings, atom_weights) = self.rna_feature(X, S, batch_id, self.k_neighbors, rna_pos ,sec_pos,chain,energys,single_embeddings,duplex_embeddings)      
        mod_h, x, _ =self.gnn(H_0, X, sirna_edges,mrna_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)
        h_ = torch.cat([mod_h[smask][0::2,:],mod_h[smask][1::2,:]],dim=1) 
        logits2=self.pred2(h_).squeeze()
        probs2 = torch.sigmoid(logits2)
        if probs2.dim() == 0:
            probs2=probs2.unsqueeze(0)
        return probs2,h_,mod_h[smask][0::2,:],mod_h[smask][1::2,:]
