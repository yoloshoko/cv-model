import torch 
from torch import nn

class channelAtt(nn.Module):
    def __init__(self,channel,reduction):
        super().__init__()
        self.max_pool = nn.AdaptiveAvgPool2d((1,1))
        self.avg_pool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Sequential(
            # mlp needs bias = False
            nn.Conv2d(channel,channel // reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction,channel,1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        b,c,w,h = x.size()
        max_res = self.max_pool(x) # b,c,1,1
        avg_res = self.avg_pool(x) # b,c,1,1
        
        max_res = self.fc(max_res) # b,c,1,1
        avg_res = self.fc(avg_res) # b,c,1,1

        res = self.sigmoid(max_res+avg_res) # b,c,1,1

        return res
    
class spatialAtt(nn.Module):
    def __init__(self,kernel):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=kernel,padding=kernel // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        max_res , _  = torch.max(x,dim=1,keepdim=True)  # b,1,w,h
        avg_res = torch.mean(x,dim=1,keepdim=True) # b,1,w,h
        res = torch.cat([max_res,avg_res],dim=1) # b,2,w,h
        res = self.conv(res) # b,1,w,h
        res = self.sigmoid(res) # b,1,w,h

        return res
    
class cbam(nn.Module):
    def __init__(self,channel,reduction,kernel):
        super().__init__()
        self.channelAtt=channelAtt(channel,reduction)
        self.spatialAtt=spatialAtt(kernel)
    def forward(self,x):
        residual = x
        out = x * self.channelAtt(x)
        out = out * self.spatialAtt(out)
        return out + residual
    
if __name__ == "__main__":
    b,c,w,h = 1,512,7,7
    x = torch.randn(b,c,w,h)

    model = cbam(channel=512,reduction=8,kernel=7)

    y = model(x)

    print(x.shape,y.shape)