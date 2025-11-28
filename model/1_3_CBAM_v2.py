import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1 , bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        max_res = self.max_pool(x)
        avg_res = self.avg_pool(x)

        max_out = self.se(max_res)
        avg_out = self.se(avg_res)

        out = max_out + avg_out

        return out
    
class spatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=kernel_size,padding= kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        max_res , _ = torch.max(x,dim=1,keepdim= True)
        avg_res = torch.mean(x,dim=1,keepdim=True)

        res = torch.cat([max_res,avg_res],dim=1)

        out = self.conv(res)

        return out
    
class cbam(nn.Module):
    def __init__(self,channel=512,reduction=16,kernel_size=49,hw=None):
        super().__init__()
        self.ChannelAttention = ChannelAttention(channel=channel,reduction=reduction)
        self.SpatialAttention = spatialAttention(kernel_size=kernel_size)
        self.joint_channel = channel + hw

        self.mlp = nn.Sequential(
            nn.Conv2d(self.joint_channel, channel // reduction , 1 ,bias= False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction , self.joint_channel , 1, bias= False)
        )

        self.sigmoid =nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        
        Channel_x = self.ChannelAttention(x).reshape(b,c,1,1)
        Spatial_x = self.SpatialAttention(x).reshape(b,h*w,1,1)

        CS_x = torch.cat([Channel_x,Spatial_x],dim=1)

        CS_xx =self.mlp(CS_x)

        Channel_x = CS_xx[:,:c,:].reshape(b,c,1,1)
        Spatial_x = CS_xx[:,c:,:].reshape(b,1,h,w)

        Channel_weight = self.sigmoid(Channel_x)
        Spatial_weight = self.sigmoid(Spatial_x)

        out1 = x * Channel_weight 
        out2 = x * Spatial_weight
        return out1,out2
    

if __name__ == "__main__":
    x =torch.randn(1,64,7,7)
    b,c,h,w = x.shape
    model = cbam(channel=64,reduction=8,kernel_size=7,hw=h*w)

    out1,out2 = model(x)
    print(x.shape,out1.shape,out2.shape)