import torch
import torch.nn as nn
import torch.nn.functional as F





import numpy 



class RNACNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(5,512)
        self.l2 = nn.Linear(167,512)
        self.l3 = nn.Linear(10,512)
        self.CNN = CNN3()

        
    def forward(self,S,CP,MASK,rnamod,pct,mode='Train'): #residue_pos → rna_pos
     
        S = self.l1(S)
        rnamod = self.l2(rnamod)
        CP = self.l3(CP)

        logits = self.CNN(rnamod,S,CP,MASK).squeeze()
        probs = torch.sigmoid(logits)
        if probs.dim() == 0:
            probs=probs.unsqueeze(0)
  

        class_loss = F.smooth_l1_loss(probs, pct.squeeze()) 

        loss=class_loss
        return loss


    def test(self,S,CP,MASK,rnamod,pct,mode='Train'):


        S = self.l1(S)
        rnamod = self.l2(rnamod)
        CP = self.l3(CP)
        logits = self.CNN(rnamod,S,CP,MASK).squeeze() 

        probs = torch.sigmoid(logits)
        if probs.dim() == 0:
            probs=probs.unsqueeze(0)

        return probs





class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim1, num_heads):
        super(CrossAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim1, num_heads=num_heads)

    def forward(self, x1, x2,mask):
        # x1: query; x2: key, value
        attn_output, _ = self.attention(x1, x2, x2,key_padding_mask=mask)
        return attn_output





class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.cross_attention = CrossAttentionModule(hidden_dim1=512,num_heads=4)   
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 128 * 12, 128) 
        self.fc2 = nn.Linear(128, 1)   
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2, x3,mask=None):
        # cross attention
 
        x1 = x1.permute(1, 0, 2)  #mod
        x2 = x2.permute(1, 0, 2)  #S
        x3 = x3.permute(1, 0, 2)  #cp
        attended1 = self.cross_attention(x1, x2,mask)
        attended1 = attended1.permute(1, 2, 0)  
        attended2 = self.cross_attention(x3, x2,mask)
        attended2 = attended2.permute(1, 2, 0)  
        attended3 = self.cross_attention(x2, x1,mask) 
        attended3 = attended3.permute(1, 2, 0)  
        # torch.Size([batch, 512, 50])

        #CNN
        x = torch.stack([attended1,attended2,attended3],dim=1)

        #([batch, 3, 512, 50]) 
        x = x.unsqueeze(dim = 2)   

        x = F.relu(self.conv1(x))
        # torch.Size([batch, 16, 512, 50])
        x = x.squeeze(dim = 2)

        x = self.maxpool(x)
        #[batch, 16, 256, 25]
        x = x.unsqueeze(dim = 2)
        x = F.relu(self.conv2(x))
        x = x.squeeze(dim = 2)
        #[128, 32, 256, 25]
        x = self.maxpool(x)
        #[128, 32, 128, 12]

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
