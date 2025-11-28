import torch 
from torch import nn 
from torch.nn import functional as F

class EMA(nn.Module):
    def __init__(self,channel,groups):
        super().__init__()
        self.groups = groups
        self.pool_h = nn.AdaptiveAvgPool2d((1,None))
        self.pool_w = nn.AdaptiveAvgPool2d((None,1))
        self.conv1x1 = nn.Conv2d(channel // groups,channel //groups,kernel_size=1, padding= 0)
        self.conv3x3 = nn.Conv2d(channel // groups,channel //groups,kernel_size=3, padding=1)

        self.gn = nn.GroupNorm(channel // groups,channel // groups)
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        b,c,h,w = x.size()
        group_x = x.reshape(b*self.groups, c // self.groups,h,w) # b*g,c/g,h,w
        x_h = self.pool_h(group_x).permute(0,1,3,2) # b*g,c/g,1,w -> b*g,c/g,w,1
        x_w = self.pool_w(group_x) # b*g,c/g,h,1
        x_hw = torch.cat([x_h,x_w],dim=2) # b*g,c/g,h+w,1
        x_hw = self.conv1x1(x_hw) # b*g,c/g,h+w,1
        x_h,x_w = torch.split(x_hw,[h,w],dim=2) # b*g,c/g,h+w,1 -> b*g,c/g,h,1 , b*g,c/g,w,1
        x_h = F.sigmoid(x_h) # b*g,c/g,h,1 
        x_w = F.sigmoid(x_w).permute(0,1,3,2) # b*g,c/g,w,1 -> b*g,c/g,1,w
        
        x1 = group_x * x_h * x_w  # b*g,c/g,h,w * b*g,c/g,h,1 * b*g,c/g,1,w -> b*g,c/g,h,w
        x2 = self.conv3x3(group_x) # b*g,c/g,h,w -> b*g,c/g,h,w

        x12 = self.gn(x1) # b*g,c/g,h,w
        x13 = self.pool_avg(x12).squeeze(-1).permute(0,2,1) # b*g,c/g,h,w -> b*g,c/g,1,1 -> b*g,c/g,1 -> b*g,1,c/g
        x13 = self.softmax(x13) # b*g,1,c/g
        x21 = x2.reshape(b*self.groups,c// self.groups,h*w) # b*g,c/g,h,w -> b*g,c/g,h*w
        y1 = torch.matmul(x13,x21) # b*g,1,c/g  @ b*g,c/g,h*w  = b*g,1,h*w

        x22 = self.pool_avg(x2) # b*g,c/g,h,w -> b*g,c/g,1,1
        x23 = self.softmax(x22.squeeze(-1).permute(0,2,1)) # b*g,c/g,1,1 -> b*g,c/g,1 -> b*g,1,c/g

        x11 = x1.reshape(b*self.groups,c// self.groups,h*w) # b*g,c/g,h,w -> b*g,c/g,h*w
        y2 = torch.matmul(x23,x11) # b*g,1,c/g @ b*g,c/g,h*w  = b*g,1,h*w

        y = y1+y2  # b*g,1,h*w
        weight = F.sigmoid(y).reshape(b*self.groups,1,h,w) # b*g,1,h*w

        # print(weight.shape,group_x.shape)
        out = weight * group_x # b*g,1,h,w @ b*g,c/g,h,w = b*g,c/g,h,w
        out = out.reshape(b,c,h,w)
        return out
    

if __name__ == "__main__":
    x = torch.randn(1,512,7,7)
    model = EMA(channel=512,groups=32)
    y = model(x)
    print(x.shape,y.shape)