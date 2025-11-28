import torch 
from torch import nn


class SKNet(nn.Module):
    def __init__(self,channel=512,reduction=8,kernels=[1,3,5,7],L=32):
        super().__init__()
        self.d = max(L,channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            # 有out_channel个卷积核
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel,channel,kernel_size=k,padding=k // 2,groups=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU()
                )
            )
        self.fc = nn.Linear(channel,self.d)
        self.softmax = nn.Softmax(dim=0)
        self.fcs = nn.ModuleList([])

        for k in kernels:
            self.fcs.append(nn.Linear(self.d,channel))    

    def forward(self,x):
        b,c,w,h = x.size()
        # b,c,w,h
        conv_outs=[]
        for conv in self.convs:
            # b,c,w,h
            out = conv(x)
            conv_outs.append(out) 
        feat = torch.stack(conv_outs,dim=0) # k,b,c,w,h

        u = sum(conv_outs) # b,c,w,h
        u = u.mean(-1).mean(-1) # b,c

        z = self.fc(u) # b,d
        
        weights=[]
        for fc in self.fcs:
            w = fc(z).view(b,c,1,1) # b,c
            weights.append(w)
        scale_weights = torch.stack(weights,dim=0) # k,b,c,1,1
        scale_weights = self.softmax(scale_weights)  # k,b,c,1,1
        out = (feat * scale_weights).sum(dim=0)  

        return out
    


if __name__ == "__main__":
    b,c,w,h = 1,512,7,7
    x = torch.randn(b,c,w,h)

    model = SKNet(channel=512,reduction=8)

    y = model(x)

    print(x.shape,y.shape)