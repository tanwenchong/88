#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_scatter import scatter_mean, scatter_sum

from utils.rna_utils import VOCAB

#from data.mod_utils import ChemUnimol_VOCAB
from data.mod_utils import MOGANRdkit_VOCAB
#from data.mod_utils import RdkitDis_VOCAB
#from data.mod_utils import MACCSRdkit_VOCAB
#from data.mod_utils import Chembert_VOCAB


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def graph_to_batch(tensor, batch_id, padding_value=0, mask_is_pad=True):
    '''
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    '''
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    batch = torch.ones((bs, max_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) * padding_value
    # generate pad mask: 1 for pad and 0 for data
    pad_mask = torch.zeros((bs, max_n + 1), dtype=torch.long, device=tensor.device)
    pad_mask[(torch.arange(bs, device=tensor.device), lengths)] = 1
    pad_mask = (torch.cumsum(pad_mask, dim=-1)[:, :-1]).bool()
    data_mask = torch.logical_not(pad_mask)
    # fill data
    batch[data_mask] = tensor
    mask = pad_mask if mask_is_pad else data_mask
    return batch, mask


def _knn_edges(X, AP, src_dst, atom_pos_pad_idx, k_neighbors, batch_info, given_dist=None):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param AP: [N, n_channel], atom position with pad type need to be ignored
    :param src_dst: [Ef, 2], full possible edges represented in (src, dst)
    :param given_dist: [Ef], given distance of edges
    '''
    offsets, batch_id, max_n, gni2lni = batch_info

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = X.shape[0]
    if given_dist is None:
        dist = X[src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = AP[src_dst] == atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * BIGINT  # [Ef, n_channel, n_channel]
        del pos_pad  # release memory
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
    else:
        dist = given_dist
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(src_dst[0], gni2lni[src_dst[1]])] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k_neighbors, dim=-1, largest=False)  # [N, topk] could >

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)

    dst = dst + offsets[batch_id[src]]  # mapping from local to global node index

    edges = torch.stack([src, dst])  # message passed from dst to src
    return edges  # [2, E]


class EdgeConstructor:
    def __init__(self, bos_idx, atom_pos_pad_idx) -> None:
        self.bos_idx=bos_idx,
        self.atom_pos_pad_idx = atom_pos_pad_idx
        

        # buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.row = None
        self.col = None
        self.row_global = None
        self.col_global = None
        self.row_seg = None
        self.col_seg = None
        self.offsets = None
        self.max_n = None
        self.gni2lni = None
        self.not_global_edges = None

    def manual_cumsum(self,tensor, dim=-1):
        # Ensure the input is a tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")
        
        # Create an empty tensor of the same shape to store the results
        result = torch.zeros_like(tensor).to(device='cuda')
        
        # Compute cumulative sum along the specified dimension
        if dim == -1:
            dim = tensor.dim() - 1
        
            # Use tensor.unbind to iterate along the specified dimension
            indices = torch.arange(tensor.size(dim), dtype=torch.long, device=tensor.device)
            for i in indices:
                if i == 0:
                    result.index_copy_(dim, torch.tensor([i], device=tensor.device), tensor.index_select(dim, torch.tensor([i], device=tensor.device)))
                else:
                    prev_sum = result.index_select(dim, torch.tensor([i-1], device=tensor.device))
                    current_val = tensor.index_select(dim, torch.tensor([i], device=tensor.device))
                    new_sum = prev_sum + current_val
                    result.index_copy_(dim, torch.tensor([i], device=tensor.device), new_sum)
        if dim == 0:
            for i in range(tensor.size(0)):
                if i == 0:
                    result[i] = tensor[i]
                else:
                    result[i] = result[i-1] + tensor[i]
        return result

    def get_batch_edges(self, batch_id):
        # construct tensors to map between global / local node index change torchscatter to torch
        ones = torch.ones_like(batch_id)

# Initialize the result tensor with zeros
        lengths = torch.zeros(batch_id.max() + 1, dtype=ones.dtype).to(device='cuda')

# Use scatter_add to perform the summation
        lengths.scatter_add_(0, batch_id, ones)
        #lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        #offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs] !!!!!!!!!!!!!

        offsets = F.pad(self.manual_cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        #same_bid = 1 - torch.cumsum(same_bid, dim=-1) !!!!!!!!!
   
        same_bid = 1 - self.manual_cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        #same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        return (row, col), (offsets, max_n, gni2lni)

    def _prepare(self, S, batch_id) -> None:
        (row, col), (offsets, max_n, gni2lni) = self.get_batch_edges(batch_id)
        
        # not global edges
        is_global = torch.tensor(S == torch.tensor(self.bos_idx).cuda()) # [N]
        
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        
        # segment ids
        

        # add to buffer
        self.row, self.col = row, col
        self.offsets, self.max_n, self.gni2lni = offsets, max_n, gni2lni
        #self.row_global, self.col_global = row_global, col_global
        self.not_global_edges = not_global_edges
        

    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: same seg, not global
        select_edges = self.not_global_edges
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges


    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_not(self.not_global_edges)
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        return global_normal

    def _construct_seq_edges(self):
        row, col = self.row, self.col
        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph
            self.not_global_edges,  # not global edges (also ensure the edges are in the same segment)
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return seq_adj

    @torch.no_grad()
    def construct_edges(self, X, S,sec_pos, batch_id, k_neighbors, atom_pos):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # prepare edge only in same small graph
        self._prepare(S, batch_id)

        ctx_edges = []

        # edges within chains
        inner_edges = self._construct_inner_edges(X, batch_id, k_neighbors, atom_pos)
        row, col = inner_edges
        inner_edges = inner_edges[:,sec_pos[row] == sec_pos[col]]

        # edges between global nodes and normal/global nodes
        global_normal = self._construct_global_edges()
        # edges on the 1D sequence
        seq_edges = self._construct_seq_edges()

        # construct  edges
        #ctx_edges = torch.cat([inner_edges,  seq_edges], dim=1) 
        ctx_edges = torch.cat([inner_edges, global_normal, seq_edges], dim=1) # [2, E]


        self._reset_buffer()
        return ctx_edges,global_normal


class GMEdgeConstructor(EdgeConstructor):
    '''
    Edge constructor for graph matching (kNN internel edges and all bipartite edges)
    '''
    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: both in ag or ab, not global

        select_edges = self.not_global_edges
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        #fconnect_edges = torch.stack([row[select_edges], col[select_edges]])
        return inner_edges #fconnect_edges#

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_not(self.not_global_edges)
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
 
        return global_normal




class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings

# embedding of nucleic acids. (default: concat rna embedding and atom embedding to one vector) residue → rna
class NucleicAcidEmbedding(nn.Module):
    '''
    [rna embedding + position embedding + modified embedding, mean(atom embeddings + atom position embeddings)]
    '''
    def __init__(self, num_rna_type, num_atom_type, num_atom_pos, rna_embed_size, atom_embed_size,
                 atom_pad_id=VOCAB.get_atom_pad_idx(), relative_position=True, max_position=51,tf_dim=512,dis_dim=7):  # max position (30+30)
        super().__init__()
        num_rna_type = 8
        self.rna_embedding = nn.Embedding(num_rna_type, rna_embed_size) #4,
        if relative_position:
            self.rna_pos_embedding = SinusoidalPositionEmbedding(rna_embed_size)  # relative positional encoding
        else:
            self.rna_pos_embedding = nn.Embedding(max_position, rna_embed_size)
            self.rna_pos_embedding35 = nn.Embedding(max_position, rna_embed_size)  # absolute position encoding
        self.atom_embedding = nn.Embedding(num_atom_type, atom_embed_size)
        self.atom_pos_embedding = nn.Embedding(num_atom_pos, atom_embed_size)
        self.chain_embedding = nn.Embedding(3, rna_embed_size)
        self.mod_embed = nn.Embedding(3, rna_embed_size)

        self.atom_pad_id = atom_pad_id
        self.eps = 1e-10  # for mean of atom embedding (some rna have no atom at all)

        #num_acid_mode,num_atom_mode=5,5
        #self.acid_mode = nn.Embedding(num_acid_mode, rna_embed_size)
        #self.atom_mode = nn.Embedding(num_atom_mode, atom_embed_size)
        self.tf_dim = tf_dim #512:unimol    384:chembert 
        self.dis_dim = dis_dim

        self.tf_to_rna = nn.Linear(self.tf_dim, rna_embed_size)

        self.mg_to_atom = nn.Linear(512, atom_embed_size)
        self.mg_to_rna = nn.Linear(512*3, rna_embed_size)
        self.ma_to_rna = nn.Linear(167, rna_embed_size)
 
        self.dis_to_rna = nn.Linear(self.dis_dim , rna_embed_size)
        self.dis_to_atom = nn.Linear(self.dis_dim , atom_embed_size)

        self.rna_oh_to_rna = nn.Linear(6 , rna_embed_size)
        self.rna_pc_to_rna = nn.Linear(10 , rna_embed_size)
        
        self.fm_embedding = nn.Linear(640,rna_embed_size)

    
    
    def forward(self, S, RP, A, AP,SM):
        '''
        :param S: [N], rna types
        :param RP: [N], rna positions
        :param A: [N, n_channel], atom types
        :param AP: [N, n_channel], atom positions
        :param SM: [N], acid mode types
        :param AM: [N, n_channel], atom mode
        '''
        rna_embed = self.rna_embedding(S) + self.rna_pos_embedding(RP) +self.acid_mode(SM)  # [N, rna_embed_size]
        atom_embed = self.atom_embedding(A) + self.atom_pos_embedding(AP)+ self.atom_mode(AM)  # [N, n_channel, atom_embed_size]
        atom_not_pad = (AP != self.atom_pad_id)  # [N, n_channel]
        denom = torch.sum(atom_not_pad, dim=-1, keepdim=True) + self.eps
        atom_embed = torch.sum(atom_embed * atom_not_pad.unsqueeze(-1), dim=1) / denom  # [N, atom_embed_size]
        return torch.cat([rna_embed, atom_embed], dim=-1)  # [N, rna_embed_size + atom_embed_size]


class NucleicAcidFeature(nn.Module):
    def __init__(self, embed_size, relative_position=True, edge_constructor=EdgeConstructor, backbone_only=False) -> None:
        super().__init__()

        self.backbone_only = backbone_only

        # number of classes
        self.num_rna_type = len(VOCAB)
        self.num_atom_type = VOCAB.get_num_atom_type()
        self.num_atom_pos = VOCAB.get_num_atom_pos()

        # atom-level special tokens
        self.atom_mask_idx = VOCAB.get_atom_mask_idx()
        self.atom_pad_idx = VOCAB.get_atom_pad_idx()
        self.atom_pos_mask_idx = VOCAB.get_atom_pos_mask_idx()
        self.atom_pos_pad_idx = VOCAB.get_atom_pos_pad_idx()
        
        # embedding


        # global nodes and mask nodes
        self.bos_idx = VOCAB.symbol_to_idx(VOCAB.BOS)

        self.mask_idx = VOCAB.get_mask_idx()

       

        # atoms encoding
        rna_atom_type, rna_atom_pos = [], [] 
        backbone = [VOCAB.atom_to_idx(atom[0]) for atom in VOCAB.backbone_atoms]
        n_channel = VOCAB.MAX_ATOM_NUMBER if not backbone_only else 3
        special_mask = VOCAB.get_special_mask()
        for i in range(len(VOCAB)):
            if i == self.bos_idx or i == self.mask_idx:
                # global nodes
                rna_atom_type.append([self.atom_mask_idx for _ in range(n_channel)])
                rna_atom_pos.append([self.atom_pos_mask_idx for _ in range(n_channel)])
            elif special_mask[i] == 1:
                # other special token (pad)
                rna_atom_type.append([self.atom_pad_idx for _ in range(n_channel)])
                rna_atom_pos.append([self.atom_pos_pad_idx for _ in range(n_channel)])
            else:
                # normal amino acids
                sidechain_atoms = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                atom_type = backbone
                atom_pos = [VOCAB.atom_pos_to_idx(VOCAB.atom_pos_bb) for _ in backbone]
                if not backbone_only:
                    sidechain_atoms = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                    atom_type = atom_type + [VOCAB.atom_to_idx(atom[0]) for atom in sidechain_atoms]
                    atom_pos = atom_pos + [VOCAB.atom_pos_to_idx(atom[1]) for atom in sidechain_atoms]
                num_pad = n_channel - len(atom_type)
                rna_atom_type.append(atom_type + [self.atom_pad_idx for _ in range(num_pad)])
                rna_atom_pos.append(atom_pos + [self.atom_pos_pad_idx for _ in range(num_pad)])

        rna_coarse = []
        for i in range(len(rna_atom_type)):
            if rna_atom_pos[i][2] == 4:
                rna_coarse.append([2,3,5])
            elif rna_atom_pos[i][2] == 3:
                rna_coarse.append([2,3,4])
            else:
                rna_coarse.append(rna_atom_pos[i])

        
        
        # mapping from rna to atom types and positions
        self.rna_atom_type = nn.parameter.Parameter(
            torch.tensor(rna_atom_type, dtype=torch.long),
            requires_grad=False)
        self.rna_atom_pos = nn.parameter.Parameter(
            torch.tensor(rna_atom_pos, dtype=torch.long),
            requires_grad=False)
        self.rna_coarse = nn.parameter.Parameter(
            torch.tensor(rna_coarse, dtype=torch.long),
            requires_grad=False)


        # edge constructor
        self.edge_constructor = edge_constructor(self.bos_idx,  self.atom_pos_pad_idx)

    def _is_global(self, S):
        return S == self.bos_idx  # [N]

    def _construct_rna_pos(self, S):
        # construct rna position. global node is 1, the first rna is 2, ... (0 for padding)
        glbl_node_mask = self._is_global(S)
        glbl_node_idx = torch.nonzero(glbl_node_mask).flatten()  # [batch_size * 3] 
        shift = F.pad(glbl_node_idx[:-1] - glbl_node_idx[1:] + 1, (1, 0), value=1) # [batch_size * 3]
        rna_pos = torch.ones_like(S)
        rna_pos[glbl_node_mask] = shift
        rna_pos = torch.cumsum(rna_pos, dim=0)
        return rna_pos


    def _construct_atom_type(self, S):
        # construct atom types
        return self.rna_atom_type[S]
    
    def _construct_atom_pos(self, S):
        # construct atom positions
        return self.rna_atom_pos[S]

    def _construct_rna_coarse(self, S):
        # construct atom types
        return self.rna_coarse[S]

    def update_globel_coordinates(self, X, S, atom_pos=None):
        X = X.clone()

        if atom_pos is None:  # [N, n_channel]
            atom_pos = self._construct_atom_pos(S)

        glbl_node_mask = self._is_global(S)
        chain_id = glbl_node_mask.long()
        chain_id = torch.cumsum(chain_id, dim=0)  # [N]
        chain_id[glbl_node_mask] = 0    # set global nodes to 0
        chain_id = chain_id.unsqueeze(-1).repeat(1, atom_pos.shape[-1])  # [N, n_channel]
        
        not_global = torch.logical_not(glbl_node_mask)
        not_pad = (atom_pos != self.atom_pos_pad_idx)[not_global]
        flatten_coord = X[not_global][not_pad]  # [N_atom, 3]
        flatten_chain_id = chain_id[not_global][not_pad]

        global_x = scatter_mean(
            src=flatten_coord, index=flatten_chain_id,
            dim=0, dim_size=glbl_node_mask.sum() + 1)  # because index start from 1
        X[glbl_node_mask] = global_x[1:].unsqueeze(1)

        return X




    def embedding(self, S, rna_pos=None, atom_type=None, atom_pos=None):
        '''
        :param S: [N], rna types
        '''
        if rna_pos is None:  # rna positions in the chain
            rna_pos = self._construct_rna_pos(S)  # [N]

        if atom_type is None:  # Atom types in each rna
            atom_type = self.rna_atom_type[S]  # [N, n_channel]

        if atom_pos is None:   # Atom position in each rna
            atom_pos = self.rna_atom_pos[S]     # [N, n_channel]

        H = self.rna_embedding(S, rna_pos, atom_type, atom_pos)
        return H, (rna_pos, atom_type, atom_pos)

    @torch.no_grad()
    def construct_edges(self, X, S,sec_pos, batch_id, k_neighbors, atom_pos=None, ):

        # prepare inputs
        if atom_pos is None:  # Atom position in each rna (pad need to be ignored)
            atom_pos = self.rna_atom_pos[S]
        

        ctx_edges = self.edge_constructor.construct_edges(
            X, S, sec_pos,batch_id, k_neighbors, atom_pos)

        return ctx_edges

    def forward(self, X, S, batch_id, k_neighbors):
        H, (_, _, atom_pos) = self.embedding(S)
        ctx_edges = self.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos=atom_pos)
        return H, ctx_edges


class SeparatedNucleicAcidFeature(NucleicAcidFeature):
    '''
    Separate embeddings of atoms and rnas
    '''
    def __init__(self, embed_size, atom_embed_size, relative_position=True, edge_constructor=EdgeConstructor, fix_atom_weights=False, backbone_only=False,feature='MACCS',atommod=False) -> None:
        super().__init__(embed_size, relative_position=relative_position, edge_constructor=edge_constructor, backbone_only=backbone_only)
        atom_weights_mask = self.rna_atom_type == self.atom_pad_idx
        self.register_buffer('atom_weights_mask', atom_weights_mask)
        self.fix_atom_weights = fix_atom_weights
        if fix_atom_weights:
            atom_weights = torch.ones_like(self.rna_atom_type, dtype=torch.float)
        else:
            atom_weights = torch.randn_like(self.rna_atom_type, dtype=torch.float)
        atom_weights[atom_weights_mask] = 0
        self.atom_weight = nn.parameter.Parameter(atom_weights, requires_grad=not fix_atom_weights)
        self.zero_atom_weight = nn.parameter.Parameter(torch.zeros_like(atom_weights), requires_grad=False)
        
        # override
        ###!!!!
        #self.chem_vocab,self.tf_dim = ChemUnimol_VOCAB
        #self.chem_vocab,self.tf_dim = Chembert_VOCAB
        self.mg_vocab,self.mg_dim = MOGANRdkit_VOCAB
        #self.dis_vocab,self.dis_dim = RdkitDis_VOCAB
        #self.ma_vocab,self.ma_dim = MACCSRdkit_VOCAB

        self.rna_embedding = NucleicAcidEmbedding(
            self.num_rna_type, self.num_atom_type, self.num_atom_pos,
            embed_size, atom_embed_size, self.atom_pad_idx, relative_position,tf_dim=0)

        self.embed_size = embed_size
        self.rnamod = True
        self.atommod = atommod
        self.feature = feature
        
        
 
    
    def get_atom_weights(self, rna_types):
        weights = torch.where(
            self.atom_weights_mask,
            self.zero_atom_weight,
            self.atom_weight
        )  # [num_rna_classes, max_atom_number(n_channel)]
        if not self.fix_atom_weights:
            weights = F.normalize(weights, dim=-1)
        return weights[rna_types]

    def forward(self, X, S, batch_id, k_neighbors, rna_pos,atom_mask,rna_raw,chain_id,mod,FM,sec_pos,mod_mask):


        atom_type = self.rna_atom_type[S]  # [N, n_channel]
        atom_pos = self.rna_atom_pos[S]     # [N, n_channel]
        rna_coarse = self.rna_coarse[S]
        #print(atom_type.shape,atom_type)
        #print(atom_pos.shape,atom_pos)



        ctx_edges,att_edges = self.construct_edges(
            X, S,sec_pos, batch_id, k_neighbors, atom_pos=atom_pos)
        

        #smask = torch.any(atom_mask != torch.tensor([0, 0, 0]).to('cuda'), dim=1)

        FM = self.rna_embedding.fm_embedding(FM)
        # rna embedding
        pos_embedding = self.rna_embedding.rna_pos_embedding(rna_pos)
        #pos_embedding35 = self.rna_embedding.rna_pos_embedding35(rna_pos35)
        #chain_embedding = self.rna_embedding.chain_embedding(chain_id)
        #rna_pp = self.rna_embedding.rna_pc_to_rna(rna_pp)
        H = self.rna_embedding.rna_embedding(S) #S

        #H = H + FM
        
        H = H + pos_embedding +FM #+rna_pp #+ chain_embedding
        
        #H = FM + pos_embedding

  


        if self.feature == 'MOGAN': #OLD MACCS
            #h_m = torch.zeros([H.shape[0],128],dtype=torch.float).to(device='cuda')
            #rnamod_embedding=torch.zeros([atom_mask[smask].shape[0],3,self.mg_dim],dtype=torch.float).to(device='cuda')

            #for i in range(atom_mask[smask].shape[0]):
            #    k=0
            #    for j in range(atom_mask[smask].shape[1]):  
            #        if  atom_mask[smask][i][j]   != 0:            
            #            rnamod_embedding[i][k] = self.mg_vocab[atom_mask[smask][i][j].item()]
            #            k+=1
            
           
            #rnamod_embedding = torch.flatten(rnamod_embedding, start_dim=1, end_dim=2)
            rnamod_embedding = torch.flatten(mod, start_dim=1, end_dim=2)
            
            rnamod_embedding = self.rna_embedding.mg_to_rna(rnamod_embedding)

            rnamod_embedding += self.rna_embedding.mod_embed(mod_mask)
            #rnamod_embedding = torch.sum(rnamod_embedding,dim=1)
            #H[smask] = H[smask] + rnamod_embedding
            #H = torch.cat([H,rnamod_embedding],dim=1)

        if self.feature == 'MACCS':
            rnamod_embedding=torch.zeros([atom_mask[smask].shape[0],3,self.ma_dim],dtype=torch.float).to(device='cuda')

            for i in range(atom_mask[smask].shape[0]):
                k=0
                for j in range(atom_mask[smask].shape[1]):  
                    if  atom_mask[smask][i][j]   != 0:            
                        rnamod_embedding[i][k] = self.ma_vocab[atom_mask[smask][i][j].item()] #self.chem_vocab
                        k+=1
            

            rnamod_embedding = torch.sum(rnamod_embedding,dim=1)
            rnamod_embedding = rnamod_embedding != 0
            rnamod_embedding = rnamod_embedding.float().to(device='cuda')
            rnamod_embedding = self.rna_embedding.ma_to_rna(rnamod_embedding) #rdkit_to_rna

            H[smask] = H[smask] + rnamod_embedding

        if self.feature == 'phychem':
            dis_embedding=torch.zeros([atom_mask[smask].shape[0],3,self.dis_dim],dtype=torch.float).to(device='cuda')

            for i in range(atom_mask[smask].shape[0]):
                k=0
                for j in range(atom_mask[smask].shape[1]):  
                    if  atom_mask[smask][i][j]   != 0:            
                        dis_embedding[i][k] = self.dis_vocab[atom_mask[smask][i][j].item()]
                        k+=1
            

            dis_embedding = self.rna_embedding.dis_to_rna(dis_embedding)
            dis_embedding = torch.sum(dis_embedding,dim=1)
            H[smask] = H[smask] + dis_embedding

        
        
        
        # atom embedding
        #atom_embedding =  self.rna_embedding.atom_pos_embedding(atom_pos) + self.rna_embedding.atom_embedding(atom_type)
        atom_embedding =   self.rna_embedding.atom_embedding(rna_coarse)
        atom_weights = self.get_atom_weights(S)

        if self.atommod == 'mogan':
            atom_mm = atom_mask!=0

            ma_embedding=torch.zeros([atom_mask[atom_mm].shape[0],self.mg_dim],dtype=torch.float).to(device='cuda')
            for i in range(atom_mask[atom_mm].size(0)):
                    ma_embedding[i] = self.mg_vocab[atom_mask[atom_mm][i].item()]

            ma_embedding=self.rna_embedding.mg_to_atom(ma_embedding)
        

            atom_embedding[atom_mm]=atom_embedding[atom_mm]+ma_embedding    

     

        if self.atommod == 'phychem':
            atom_mm = atom_mask!=0
            dis_embedding=torch.zeros([atom_mask[atom_mm].shape[0],self.dis_dim],dtype=torch.float).to(device='cuda')
            for i in range(atom_mask[atom_mm].size(0)):
                    dis_embedding[i] = self.dis_vocab[atom_mask[atom_mm][i].item()]

            dis_embedding=self.rna_embedding.dis_to_atom(dis_embedding)
        

            atom_embedding[atom_mm]=atom_embedding[atom_mm]+dis_embedding        
        
        

        #return H, (ctx_edges,att_edges), (atom_embedding, atom_weights)
        return H,rnamod_embedding,(ctx_edges,att_edges), (atom_embedding, atom_weights)

class EasyNucleicAcidFeature(NucleicAcidFeature):
    '''
    Separate embeddings of atoms and rnas
    '''
    def __init__(self, embed_size, atom_embed_size, relative_position=True, edge_constructor=EdgeConstructor, fix_atom_weights=False, backbone_only=False) -> None:
        super().__init__(embed_size, relative_position=relative_position, edge_constructor=edge_constructor, backbone_only=backbone_only)
        atom_weights_mask = self.rna_atom_type == self.atom_pad_idx
        self.register_buffer('atom_weights_mask', atom_weights_mask)
        self.fix_atom_weights = fix_atom_weights
        if fix_atom_weights:
            atom_weights = torch.ones_like(self.rna_atom_type, dtype=torch.float)
        else:
            atom_weights = torch.randn_like(self.rna_atom_type, dtype=torch.float)
        atom_weights[atom_weights_mask] = 0
        self.atom_weight = nn.parameter.Parameter(atom_weights, requires_grad=not fix_atom_weights)
        self.zero_atom_weight = nn.parameter.Parameter(torch.zeros_like(atom_weights), requires_grad=False)
        
        # override
        self.chem_vocab,self.tf_dim = ChemRdkit_VOCAB

        self.rna_embedding = NucleicAcidEmbedding(
            self.num_rna_type, self.num_atom_type, self.num_atom_pos,
            embed_size, atom_embed_size, self.atom_pad_idx, relative_position,tf_dim=self.tf_dim)

        self.embed_size = embed_size
        
        


    def forward(self, X, S, batch_id, k_neighbors, rna_pos,atom_mask):
        


        smask = torch.any(atom_mask != torch.tensor([0, 0, 0]).to('cuda'), dim=1)
        
        atom_mm = atom_mask!=0

        if rna_pos is None:
            rna_pos = self._construct_rna_pos(S)  # [N]
        atom_type = self.rna_atom_type[S]  # [N, n_channel]
        atom_pos = self.rna_atom_pos[S]     # [N, n_channel]

        # rna embedding
        pos_embedding = self.rna_embedding.rna_pos_embedding(rna_pos)
        H = self.rna_embedding.rna_embedding(S)

    
        rnamod_embedding=torch.zeros([atom_mask[smask].shape[0],3,self.tf_dim],dtype=torch.float).to(device='cuda')

        for i in range(atom_mask[smask].shape[0]):
            k=0
            for j in range(atom_mask[smask].shape[1]):  
                if  atom_mask[smask][i][j]   != 0:            
                    rnamod_embedding[i][k] = self.chem_vocab[atom_mask[smask][i][j].item()]
                    k+=1
            
            #rnamod_embedding = self.rna_embedding.tf_to_rna(rnamod_embedding)
        rnamod_embedding = self.rna_embedding.rdkit_to_rna(rnamod_embedding)
        rnamod_embedding = torch.sum(rnamod_embedding,dim=1)
        H[smask] = H[smask] + rnamod_embedding

        H = H + pos_embedding 


        


        
        ctx_edges = self.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos=atom_pos)
       
        return H, ctx_edges


class CoordNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor(0)
        self.std = torch.tensor(10)
        self.mean = nn.parameter.Parameter(self.mean, requires_grad=False)
        self.std = nn.parameter.Parameter(self.std, requires_grad=False)
        self.bos_idx = VOCAB.symbol_to_idx(VOCAB.BOS)

    def normalize(self, X):
        X = (X - self.mean) / self.std
        return X

    def unnormalize(self, X):
        X = X * self.std + self.mean
        return X

    def centering(self, X, S, batch_id, rna_feature: NucleicAcidFeature):
        X = rna_feature.update_globel_coordinates(X, S)
        self.centers = X[S == self.bos_idx][:, 0]
        centers = torch.zeros(X.shape[0], X.shape[-1], dtype=X.dtype, device=X.device)
        centers = self.centers[batch_id]
        X = X - centers.unsqueeze(1)
        
        return X



    def clear_cache(self):
        self.centers = None
