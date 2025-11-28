import torch 
from torch import nn 
import numpy as np
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v,h,dropout=0.1):
        super().__init__()
        self.fc_q = nn.Linear(d_model,d_k * h)
        self.fc_k = nn.Linear(d_model,d_k * h)
        self.fc_v = nn.Linear(d_model,d_v * h)
        self.fc_o = nn.Linear(d_v * h,d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.dropout = nn.Dropout(dropout)
    def forward(self,q,k,v):
        b,n,c = q.size()
        qx = self.fc_q(q).reshape(b,n,self.h,self.d_k).permute(0,2,1,3) # b,h,n,d_k
        kx = self.fc_k(k).reshape(b,n,self.h,self.d_k).permute(0,2,3,1) # b,h,d_k,n
        vx = self.fc_v(v).reshape(b,n,self.h,self.d_v).permute(0,2,1,3) # b,h,n_d,k

        att = torch.matmul(qx,kx) / np.sqrt(self.d_k) # b,h,n,n
        att = torch.softmax(att,dim=-1) # b,h,n,n
        att = self.dropout(att)

        # b,h,n,d_k -> b,n,h,d_k -> b,n,-1
        out = torch.matmul(att,vx).permute(0,2,1,3).reshape(b,n,-1)
        out = self.fc_o(out) # b,n,d_model
        
        return out
    


if __name__ == "__main__":
    # b,n,c
    x = torch.randn(2,50,64)
    model = ScaledDotProductAttention(d_model=64,d_k=64,d_v=64,h=8)

    y = model(x,x,x)
    print(x.shape,y.shape)