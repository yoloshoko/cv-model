import torch 
from torch import nn 

class ECA(nn.Module):
    def __init__(self,kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv1d(1,1,kernel_size=kernel_size,padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y = self.pool(x) # b,c,1,1
        y = y.squeeze(-1).permute(0,2,1) # b,c,1 -> b,1,c
        y = self.conv(y) # b,1,c -> b,1,c
        y = self.sigmoid(y).unsqueeze(-1).permute(0,2,1,3) # b,1,c -> b,1,c,1
        # print(y.shape,type(y))

        return x * y.expand_as(x)
    
if __name__ == "__main__":
    x = torch.randn(1,512,7,7)
    model =ECA(kernel_size=3)
    y = model(x)
    print(x.shape,y.shape)