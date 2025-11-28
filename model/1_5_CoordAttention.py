import torch 
from torch import nn

class CoorAttention(nn.Module):
    def __init__(self,inp,outp):
        super().__init__()
        self.pool_x = nn.AdaptiveAvgPool2d((None,1))
        self.pool_y = nn.AdaptiveAvgPool2d((1,None))
        
        self.reduction = 8
        self.conv = nn.Conv2d(inp, inp // 8,kernel_size=1 , stride=1, padding=0)
        self.bn = nn.BatchNorm2d(inp // 8) 
        self.relu = nn.ReLU()

        self.conv_h = nn.Conv2d(inp // 8, outp, kernel_size=1 ,stride=1 ,padding=0)
        self.conv_w = nn.Conv2d(inp // 8, outp, kernel_size=1 ,stride=1 ,padding=0)
        
    def forward(self,x):
        b,c,h,w  = x.size()
        residual = x
        px = self.pool_x(x)  # b,c,h,1
        py = self.pool_y(x).permute(0,1,3,2)  # b,c,w,1
        
        y = torch.cat([px,py],dim=2) # b,c,(h+w),1
        y = self.conv(y) # b,d,(h+w),1
        y = self.bn(y)  # b,d,(h+w),1
        y = self.relu(y)

        px, py =torch.split(y,[h,w],dim=2) # b,d,h,1 / b,d,w,1

        ax = self.conv_h(px).sigmoid() # b,c,h,1
        ay = self.conv_w(py).sigmoid().permute(0,1,3,2) # b,c,1,w

        out = x * ax * ay
        return out
    
if __name__ == "__main__":
    x = torch.randn(1,512,7,7)
    model = CoorAttention(512,512)

    y = model(x)
    print(x.shape,y.shape)    
