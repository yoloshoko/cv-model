import torch 
from torch import nn

class SENet(nn.Module):
    def __init__(self,channel,reduction):
        super().__init__()
        self.channel = channel
        self.reduction = reduction

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel,channel // reduction,bias= False),
            nn.ReLU(),
            #? required channel * 3
            nn.Linear(channel // reduction, channel * 3,bias=False)
        )
    def forward(self,x1,x2,x3):
        b,c,w,h = x1.size()
        x =  x1 + x2 + x3
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,3*c,1,1)

        #? required torch.sigmoid
        w1 = torch.sigmoid(y[:,:c,...])
        w2 = torch.sigmoid(y[:,c:2*c,...])
        w3 = torch.sigmoid(y[:,2*c:,...])

        out = w1 *x1 + w2*x2 + w3*x3
        return out

if __name__=="__main__":
    b,c,w,h=1,512,7,7
    x1 = torch.randn(b,c,w,h)
    x2 = torch.randn(b,c,w,h)
    x3 = torch.randn(b,c,w,h)

    model = SENet(channel=512,reduction=8)

    y = model(x1,x2,x3)
    print(x1.shape,y.shape)

