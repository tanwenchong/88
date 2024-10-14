import torch
import math
import copy
import torch.nn as nn
from torch.autograd import Variable
import os
#os.environ['CUDA_VISIBLE_dEVICES']='6,7'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import pdb

    
class PositionalEncoding(nn.Module):
    def __init__(self, dim, DEVICE, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim, device=DEVICE)
        position = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,dim,2,device=DEVICE) * (-math.log(10000.0) / dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return x


class RNAFM_SIPRED_2(nn.Module):
    def __init__(self, dp=0.1, device=None):
        super(RNAFM_SIPRED_2, self).__init__()

        
        self.flat = nn.Flatten()

        """ FOR sirna """
        self.x_map = nn.Linear(640, 16)
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=16, nhead=2, dropout=dp) #  [b, len, dim]
        self.encoder = nn.TransformerEncoder(self.encoder_layer_1, num_layers=4)
        self.pos_embed = PositionalEncoding(dim=16, dropout=dp, max_len=1000, DEVICE=device)
        self.y_map = nn.Linear(16, 4)

        """ for mrna """
        self.x_map_2 = nn.Linear(640, 8)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=8, nhead=1, dropout=dp) #  [b, len, dim]
        self.encoder_2 = nn.TransformerEncoder(self.encoder_layer_2, num_layers=2)
        self.pos_embed_2 = PositionalEncoding(dim=8, dropout=dp, max_len=1000, DEVICE=device)
        self.y_map_2 = nn.Linear(8, 1)


        self.DNN_in_c = 19 * 4 + 59 * 1 + 110 #from 21 to 19

        self.bn = nn.BatchNorm1d(self.DNN_in_c)
        self.DNN = nn.Sequential(
            nn.Linear(self.DNN_in_c, 256),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.init_weights()

    def init_weights(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(self.weight.data)
            nn.init.constant_(self.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(self.weight.data)
            nn.init.constant_(self.bias.data, 0.0)


    def forward(self, inputs):

        x = inputs['rnafm_encode'] # [b, 21, 640]
        x = self.x_map(x) # [b, 21, 8]
        x = x.permute((1,0,2)) # [21, b, 8]  注意力输入为 l, b, d
        x = self.pos_embed(x)
        x = self.encoder(x)
        x = x.permute((1,0,2)) # [b, 21, 16]
        x = self.y_map(x) 
        as_att_out = self.flat(x)

        x = inputs['rnafm_encode_mrna']
        x = self.x_map_2(x)
        x = x.permute((1,0,2))
        x = self.pos_embed_2(x)
        x = self.encoder_2(x)
        x = x.permute((1,0,2))
        x = self.y_map_2(x) 
        mrna_att_out = self.flat(x)

        other_out = torch.cat([
            inputs['sirna_gibbs_energy'].to(torch.float32),  # b,20
            inputs['pssm_score'].to(torch.float32),  # b,1
            inputs['gc_sterch'].to(torch.float32),  # b,1
            inputs['sirna_second_percent'].to(torch.float32),  # b,2
            inputs['sirna_second_energy'].to(torch.float32),  # b,1
            inputs['tri_nt_percent'].to(torch.float32),  # b,64
            inputs['di_nt_percent'].to(torch.float32),  # b,16
            inputs['single_nt_percent'].to(torch.float32),  # b,4
            inputs['gc_content'].to(torch.float32),  # b,1
        ], dim=1).squeeze(2)  # b, ALL 116

        concat_out = torch.cat([as_att_out, mrna_att_out, other_out], dim=1)
        out = self.bn(concat_out)
        out = self.DNN(out)

        return out.squeeze(1)  # b,1