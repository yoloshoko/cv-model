import torch
from torch import nn 
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v,h):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k 
        self.d_v = d_v
        self.h = h
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v,d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self,q,k,v):
        # b,n,c
        # print(q.shape)
        b,n,c = q.size()
        q_x = self.fc_q(q).view(b,n,self.h,self.d_k).permute(0,2,1,3) # b,n,h*d_k -> b,n,h,d_k -> b,h,n,d_k
        k_x = self.fc_k(k).view(b,n,self.h,self.d_k).permute(0,2,3,1) # b,n,h*d_k -> b,n,h,d_k -> b,h,d_k,n
        v_x = self.fc_v(v).view(b,n,self.h,self.d_v).permute(0,2,1,3) # b,n,h*d_v -> b,n,h,d_v -> b,h,n,d_v
        att = torch.matmul(q_x,k_x) / np.sqrt(self.d_k)  # b,h,n,n
        att = torch.softmax(att,dim=-1) # b,h,n,n
        att = self.dropout(att)

        # b,h,n,n * b,h,n,d_v -> b,h,n,d_v -> b,n,h*d_v
        # print(att.shape,v_x.shape)
        out = torch.matmul(att,v_x).permute(0,2,1,3).view(b,n,-1)
        # b,n,h*d_v -> b,n,c
        # print(out.shape)
        out = self.fc_o(out)
        return out

class PositionAttention(nn.Module):
    def __init__(self,d_model=512,kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(d_model,d_model,kernel_size= kernel_size,padding= kernel_size // 2)
        self.pos_att = ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    def forward(self,x):
        b,c,h,w = x.size()
        y = self.conv(x) # b,c,h,w
        y = y.reshape(b,c,-1).permute(0,2,1) # b,c,h*w -> b,c,n -> b,n,c
        y = self.pos_att(y,y,y) # b,n,c —> b,n,c
        return y
    
class ChannelAttention(nn.Module):
    def __init__(self,d_model=512,kernel_size=3,h=7,w=7):
        super().__init__()
        self.conv = nn.Conv2d(d_model,d_model,kernel_size= kernel_size,padding= kernel_size // 2)
        self.cha_att = ScaledDotProductAttention(h*w,d_k=h*w,d_v=h*w,h=1)
    def forward(self,x):
        b,c,h,w = x.size()
        y = self.conv(x) # b,c,h,w
        y = y.reshape(b,c,-1) # b,c,h,w -> b,c,n
        y = self.cha_att(y,y,y) # b,c,n -> b,c,n
        return y

class DANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cAtt = ChannelAttention()
        self.pAtt = PositionAttention()
    def forward(self,x):
        b,c,h,w = x.size()
        p_out = self.pAtt(x) # b,n,c
        p_out = p_out.permute(0,2,1).view(b,c,h,w) # b,n,c -> b,c,n -> b,c,h,w
        
        c_out = self.cAtt(x) # b,c,n
        c_out = c_out.view(b,c,h,w)

        out = x + p_out * 0.5
        out = out + c_out * 0.2
        return out 
    

if __name__ == "__main__":
    x = torch.randn(1,512,7,7)
    model = DANet()
    y = model(x)
    print(x.shape,y.shape)