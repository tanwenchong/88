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
#from model.mt_egnn import AMEGNN
from model.mc_egnn import MCAttEGNN
from model.gcn_gat import GCN,GAT
from torch_geometric.nn import global_mean_pool,global_max_pool

class RNAmaskModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9,
                 n_layers=3,  dropout=0.1,MLM=False) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        atom_embed_size = 16
        
        self.rna_feature = SeparatedNucleicAcidFeature(
            int(embed_size/2), atom_embed_size, #embed_size
            relative_position=True,
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
        #print(mask_ratio)
        #marker_mask = marker != -0.01
      
      
        #扰动位置############################################################
        not_global = smask == 0
        rand_mask = torch.rand_like(smask, dtype=torch.float) <= mask_ratio # 1 == no mask
        is_mask = rand_mask  &  not_global
        #####################################################################


        H_0, rnamod_embedding,(ctx_edges,att_edges), (atom_embeddings, atom_weights) = self.rna_feature(X, S, batch_id, self.k_neighbors, rna_pos ,atom_mask , rna_raw,chain_id,mod,FM,sec_pos,mod_mask)

        row, col = ctx_edges

        edge_mask = torch.zeros([ctx_edges.size(1)],dtype=torch.long).cuda()

        edge_mask[sec_pos[row] == sec_pos[col]] = 1


        #坐标扰动############################################################
        if mask_ratio != 0:
            noise_level = 0.01
            noise = torch.randn_like(X[is_mask]) * noise_level
            #X[is_mask] = X[is_mask] + noise
        ##################################################################### 
  
   
        #h, mod_h=self.gnn(H_0, rnamod_embedding,X, ctx_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)
        mod_h, x, _ = self.gnn(torch.cat([H_0,rnamod_embedding],dim=-1), X, ctx_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)#torch.cat([H_0,rnamod_embedding],dim=-1)
        #mod_h, x, _ = self.gnn(H_0+rnamod_embedding, X, ctx_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)
        #mod_h = self.gnn(torch.cat([H_0,rnamod_embedding],dim=-1),ctx_edges)
        #cc = self.c(cc.unsqueeze(1))
        #cc = cc.unsqueeze(1)
        #h_ = global_mean_pool(mod_h, batch_id)
        h_ = mod_h[smask]
        #h_ = torch.cat([mod_h[smask],cc],dim=1)
        #h_ = mod_h[smask] #* cc
        #h_ = torch.cat([global_h,cc],dim=1)  

        
        #logits1=self.pred1(h[smask]).squeeze()
        logits2=self.pred2(h_).squeeze()
        #logits3=self.pred3(mod_h[smask]-h[smask]).squeeze()

        #probs1 = torch.sigmoid(logits1)
        probs2 = torch.sigmoid(logits2)
        #probs3 = torch.tanh(logits3)

        if probs2.dim() == 0:
            #probs1=probs1.unsqueeze(0)
            probs2=probs2.unsqueeze(0)
            #probs3=probs3.unsqueeze(0)

        #h_loss = F.smooth_l1_loss(probs1, marker) #/ marker_mask.sum() #[marker_mask]

        mod_loss = F.smooth_l1_loss(probs2, pct) #/ smask.sum()

        #diff_loss = F.smooth_l1_loss(probs3, pct - marker) #/ marker_mask.sum()



       
        #loss = h_loss + mod_loss + diff_loss
        loss = mod_loss
        return loss


        



    def test(self,S,X,rna_pos,sec_pos,lengths,pct,marker,smask,atom_mask,mod_mask,rna_raw,chain_id,mod,cc,FM):
   
        

        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        
        H_0, rnamod_embedding,(ctx_edges,att_edges), (atom_embeddings, atom_weights) = self.rna_feature(X, S, batch_id, self.k_neighbors, rna_pos ,atom_mask , rna_raw,chain_id,mod,FM,sec_pos,mod_mask)

        row, col = ctx_edges
        

        edge_mask = torch.zeros([ctx_edges.size(1)],dtype=torch.long).cuda()

        edge_mask[sec_pos[row] == sec_pos[col]] = 1
        
  

                
        
        mod_h, x, mid_h =self.gnn(torch.cat([H_0,rnamod_embedding],dim=-1), X, ctx_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)
        #h, mod_h, x =self.gnn(H_0,rnamod_embedding, X, ctx_edges,channel_attr=atom_embeddings,channel_weights=atom_weights)
        #cc = self.c(cc.unsqueeze(1))
        #cc = cc.unsqueeze(1)
        #h_ = torch.cat([mod_h[smask],cc],dim=1) 
        #h_ = global_mean_pool(mod_h, batch_id)
        h_ = mod_h[smask] #* cc

        #logits1=self.pred1(h[smask]).squeeze()
        logits2=self.pred2(h_).squeeze()
        #logits2 = logits2 * cc
        #logits3=self.pred3(mod_h[smask]-h[smask]).squeeze()

        #probs1 = torch.sigmoid(logits1)
        probs2 = torch.sigmoid(logits2)
        #probs3 = torch.tanh(logits3)

        if probs2.dim() == 0:
            #probs1=probs1.unsqueeze(0)
            probs2=probs2.unsqueeze(0)
            #probs3=probs3.unsqueeze(0)
        return probs2,H_0

if __name__ == '__main__':
    torch.random.manual_seed(0)
    # equivariance test
    embed_size, hidden_size = 128, 256
    n_channel, d = 3, 3
    scale = 10
    dtype = torch.float
    device = torch.device('cuda:0')
    model = RNAmaskModel(embed_size, hidden_size, n_channel,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   k_neighbors=9, n_layers=3)
    model.to(device)
    model.eval()

    ag_len=42
    center_x = torch.randn(3, 1, n_channel, d, device=device, dtype=dtype) * scale
    ag_X = torch.randn(ag_len, n_channel, d, device=device, dtype=dtype) * scale
    


    X = torch.cat([center_x[0], ag_X], dim=0)
    X1 = X
    S = torch.cat([torch.tensor([model.bos_idx], device=device),
                   torch.randint(low=0, high=3, size=(ag_len,), device=device),], dim=0)
    cmask = torch.tensor([0] + [1 for _ in range(ag_len)], device=device).bool()
    smask = torch.zeros_like(cmask)
    smask[ag_len+10:ag_len+20] = 1
    rna_pos = torch.tensor(([0] + [i+1 for i in range(ag_len)]), device=device)

    lengths = torch.tensor([ag_len +1], device=device)
    mod = torch.randn(ag_len+1, 512,3, device=device, dtype=dtype)

    torch.random.manual_seed(1)
    _,gen_X = model.test(S,X,rna_pos,None,lengths,None,None,smask,None,None,None,mod,None,None)
    # tmpx1 = model.tmpx

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q_ag, t_ag = U.mm(V), torch.randn(3, device=device)



    ag_X = torch.matmul(ag_X, Q_ag) + t_ag
    X = torch.cat([center_x[0], ag_X], dim=0)
    X2 = X


    # this is f([Q1x1+t1, Q2x2+t2])
    torch.random.manual_seed(1)

    _,gen_op_X = model.test(S,X,rna_pos,None,lengths,None,None,smask,None,None,None,mod,None,None)
    # tmpx2 = model.tmpx
    # gt_tmpx = torch.matmul(tmpx1, Q_ag) + t_ag
    # error = torch.abs(gt_tmpx[:, :4] - tmpx2[:, :4]).sum(-1).flatten().mean()
    # # error = torch.abs(tmpx2 - tmpx1).sum(-1).flatten().mean()
    # print(error.item())
    # assert error < 1e-3
    # print('independent equivariance check passed')


    gt_op_X = torch.matmul(gen_X, Q_ag) + t_ag
    #gt_op_X = torch.matmul(X1, Q_ag) + t_ag
    #gen_op_X = X2

    error = torch.abs(gt_op_X[cmask][:, :3] - gen_op_X[cmask][:, :3]).sum(-1).flatten().mean()
    #print(X)
    #print(gen_X)
    #print(gen_op_X)
    print(error.item())
    assert error < 1e-3
    print('independent equivariance check passed')