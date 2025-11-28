import torch 
from torch import nn  

class TemporalAttention(nn.Module):
    def __init__(self,window_size=3,dim=2,num_heads=1,seq_len=12):
        super().__init__()
        self.window_size = window_size
        self.qkv = nn.Linear(dim, 3 * dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.mask = torch.tril(torch.ones(window_size,window_size))
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(0.1)
    def forward(self,x): 
        # b*n,t,c
        b_, t_, c_ = x.size()
        
        x = x.reshape(-1,self.window_size,c_) # b*n,t,c -> b*n*t/ws,ws,c

        b,t,c = x.shape
        qkv = self.qkv(x).reshape(b,t,3,self.num_heads, c // self.num_heads)  # B,T,C -> B,T,3*C -> B,T,3,h,d 
        qkv = qkv.permute(2,0,3,1,4) # B,T,3,h,d -> 3,B,h,T,d 

        q, k, v = qkv[0], qkv[1], qkv[2] # B,h,T,d ; B,h,T,d ; B,h,T,d
        att = (q @ k.transpose(-2,-1)) * self.scale  # B,h,T,d @ B,h,d,T = B,h,T,T 

        att = att.masked_fill_(self.mask == 0, float("-inf"))
        att = att.softmax(dim=-1)
        
        x = (att @ v).transpose(1,2).reshape(b,t,c) # B,h,T,T @ B,h,T,d = B,h,T,d -> B,T,h,d

        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(b_,t_,c_)

        return x

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,x):
        self.fn(self.norm(x))
        return x

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(p=dropout)
        )
    def forward(self,x):
        return x

class ctMsa(nn.Module):
    def __init__(self,seq_len=12,cha_len=2,window_size=3):
        super().__init__()
        self.pos_embedding = nn.Parameter( torch.randn(seq_len,cha_len) )
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleList([
                TemporalAttention(window_size=window_size),
                PreNorm(dim=cha_len,fn=FeedForward(cha_len,hidden_dim=32))
            ])
        )

    def forward(self,x):
        b,n,t,c = x.size()
        x = x.reshape(b*n,t,c)
        x = x + self.pos_embedding

        for attn,ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b,n,t,c)
        return x

if __name__ == "__main__":
    # b,n,c,t
    x = torch.randn(64,50,12,2)
    modules = nn.ModuleList()

    blocks = 3
    seq_len = 12
    for b in range(blocks): 
        # 12 // 2 ** 2 = 3, 12 // 2 ** 1 =6, 12 // 2 ** 0 =12
        window_size = seq_len // 2 **(blocks - 1 - b)   
        modules.append(ctMsa(window_size=window_size))

    y = x
    for i in range(blocks):
        y = modules[i](y)

    print(x.shape,y.shape)