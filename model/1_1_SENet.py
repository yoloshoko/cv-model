import torch 
from torch import nn

#? required nn.Module
class SENet(nn.Module):
    def __init__(self,channel,reduction):
        #? required super().__init__()
        super().__init__()
        self.channel=channel
        self.reduction=reduction

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            #? required bias=False
            nn.Linear(channel,channel // reduction,bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,w,h=x.size()
        # b,c,1,1
        y = self.avg_pool(x).view(b,c)
        # b,c
        y = self.fc(y).view(b,c,1,1)
        out = x * y
        return out
    

if __name__=="__main__":
    x = torch.randn(1,512,7,7)
    model = SENet(channel=512,reduction=8)
    y = model(x)
    print(x.shape,y.shape)