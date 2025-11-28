import torch
from torch import nn 
from torch.nn import functional as F

class CoTAttention(nn.Module):
    def __init__(self,channel=512,kernel_size=3):
        super().__init__()
        self.key_embed = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=kernel_size,padding= kernel_size // 2,bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.kernel_size = kernel_size
        factor=4
        self.conv = nn.Sequential(
            nn.Conv2d(2* channel,2*channel // factor,1,bias=False),
            nn.BatchNorm2d(2*channel // factor),
            nn.ReLU(),
            nn.Conv2d(2* channel // factor, kernel_size * kernel_size * channel,1,bias=False),
        )

        self.value_embed = nn.Sequential(
            nn.Conv2d(channel,channel,1,bias=False),
            nn.BatchNorm2d(channel),
            # nn.ReLU()
        )
    def forward(self,x):
        b,c,h,w = x.size()
        k1 = self.key_embed(x) # b,c,h,w

        y = torch.cat([k1,x],dim=1) # b,2c,h,w
        y = self.conv(y) # b,k*k*c,h,w

        v = self.value_embed(x).view(b,c,-1) # b,c,h,w -> b,c,h*w

        y = y.view(b,c,self.kernel_size * self.kernel_size,h,w) # b,k*k*c,h,w -> b,c,k*k,h,w 
        y = torch.mean(y, dim=2 ,keepdim=False).view(b,c,-1) # b,c,k*k,h,w -> b,c,h,w -> b,c,h*w
        att = F.softmax(y,dim=-1)  # b,c,h*w
        # print(att.shape,v.shape)
        k2 = att * v 
        # print(k2.shape)
        k2 = k2.view(b,c,h,w)
        return k1+k2
    

if __name__ == "__main__":
    x = torch.randn(1,512,7,7)
    model = CoTAttention()
    y = model(x)

    print(x.shape,y.shape)