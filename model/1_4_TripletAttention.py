import torch 
from torch import nn

class BasicConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=7, padding= 7 // 2,stride=1)
    def forward(self,x):
        out = self.conv(x)
        return out
    

class ZPool(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        # b,w,c,h -> b,1,c,h
        max_res = torch.max(x,dim=1,keepdim=True)[0]
        avg_res = torch.mean(x,dim=1,keepdim=True)
        out = torch.cat([max_res,avg_res],dim=1) # b,2,c,h



        # print(f"zpool:{out.shape}")
        return out
    
class AttentionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.zpool = ZPool()
        self.conv = BasicConv(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x_pool = self.zpool(x) # b,2,c,h
        x_conv = self.conv(x_pool) # b,1,c,h
        out = self.sigmoid(x_conv) # b,1,c,h
        out = x * out # b,w,c,h * b,1,c,h = b,w,c,h
        return out
    

class TripletAttention(nn.Module):
    def __init__(self,no_spatial=False):
        super().__init__()
        self.cwAtt = AttentionGate()
        self.chAtt = AttentionGate()

        self.no_spatial = no_spatial
        if not no_spatial:
            self.hwAtt = AttentionGate()
    def forward(self,x):
        # b,c,h,w
        x1 = x.permute(0,2,1,3).contiguous() # b,h,c,w
        out1 = self.cwAtt(x1) # b,h,c,w
        out1 = out1.permute(0,2,1,3).contiguous() # b,c,h,w

        x2 = x.permute(0,3,2,1).contiguous() # b,w,h,c
        out2 = self.chAtt(x2) # b,w,c,h
        out2 = out2.permute(0,3,2,1).contiguous() # b,c,h,w



        # print(out1.shape,out2.shape)
        if not self.no_spatial:
            out3 = self.hwAtt(x)
            out = (out1 + out2 + out3) /3
        else:
            out = (out1 + out2) /2
        return out
    

if __name__ == "__main__":
    x = torch.randn(1,512,7,7)

    model =TripletAttention(no_spatial=False)
    
    y = model(x)

    print(x.shape,y.shape)