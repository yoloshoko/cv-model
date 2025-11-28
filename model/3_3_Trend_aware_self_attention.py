import torch 
from torch import nn  
from torch.nn import functional as F

class Trend_aware_attention(nn.Module):
    def __init__(self,kernel_size=3,in_dim=2,hid_dim=32,out_dim=1,his_num=12,pred_num=6):
        super().__init__()
        self.k=4
        self.d=8
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.cnn_q = nn.Conv2d(in_dim,hid_dim,(1,kernel_size),padding=(0,kernel_size - 1))
        self.cnn_k = nn.Conv2d(in_dim,hid_dim,(1,kernel_size),padding=(0,kernel_size - 1))
        # self.cnn_v = nn.Conv2d(in_dim,hid_dim,(1,kernel_size),padding=(0,kernel_size - 1))
        self.norm_q = nn.BatchNorm2d(hid_dim)
        self.norm_k = nn.BatchNorm2d(hid_dim)
        self.fc_v = nn.Linear(in_dim,hid_dim)
        self.fc_1 = nn.Linear(hid_dim,out_dim)
        self.fc_2 = nn.Linear(his_num,pred_num)
    def forward(self,x):
        b,n,t,c = x.size()
        redisual = x
        x = x.reshape(b,c,n,t) # b,n,t,c -> b,c,n,t
        
        q = self.norm_q(self.cnn_q(x))[:,:,:,:-self.padding].permute(0,3,2,1)  # b,c,n,t -> b,c_,n,t_ -> b,t,n,c_
        k = self.norm_k(self.cnn_k(x))[:,:,:,:-self.padding].permute(0,3,2,1) # b,c,n,t -> b,c_,n,t_ -> b,t,n,c_
        v = self.fc_v(redisual).permute(0,2,1,3) # b,n,t,c -> b,n,t,c_ -> b,t,n,c_

        q = torch.cat(torch.split( q,self.d,dim=-1 ),dim=0) # b*k,t,n,d   c_=k*d
        k = torch.cat(torch.split( k,self.d,dim=-1 ),dim=0) # b*k,t,n,d   c_=k*d
        v = torch.cat(torch.split( v,self.d,dim=-1 ),dim=0) # b*k,t,n,d   c_=k*d

        q = q.permute(0,2,1,3) # b*k,n,t,d
        k = k.permute(0,2,3,1) # b*k,n,d,t
        v = v.permute(0,2,1,3) # b*k,n,t,d

        att = (q @ k) * (self.d ** -0.5) # b*k,n,t,t
        att = F.softmax(att, dim = -1)   # b*k,n,t,t
        x = (att @ v)  # b*k,n,t,t @ b*k,n,t,d = b*k,n,t,d
        x = torch.cat( torch.split(x,b,dim=0),dim=-1) # b,n,t,c_

        x = self.fc_1(x).squeeze(-1) # b,n,t,c_ -> b,n,t,1 -> b,n,t
        x = self.fc_2(x) # b,n,t -> b,n,t'

        return x


if __name__ == "__main__":

    x = torch.randn(64,50,12,2)
    model = Trend_aware_attention(kernel_size=3)
    y = model(x)
    print(x.shape,y.shape)