import torch 
from torch import nn 
from torch.nn import functional as F

def get_mask(input_size=12,window_size=[2,2,3],inner_size=3):
    all_size = []
    all_size.append(input_size)

    for i in range(len(window_size)):
        layer_size = all_size[i] // window_size[i]
        all_size.append(layer_size)

    # print(all_size)
    seq_length = sum(all_size)
    mask = torch.zeros(seq_length,seq_length)

    inner_window = inner_size // 2
    # 组内的相关性
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start,start + all_size[layer_idx]):
            left_side = max(i - inner_window,start)
            right_side = min(i + inner_window + 1,start + all_size[layer_idx])
            if layer_idx == 0:
                mask[i,0:right_side]=1
            else:
                mask[i,left_side:right_side]=1

    # 组间的相关性
    for layer_idx in range(1,len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start,start+all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1]  
            if i == (start+all_size[layer_idx] -1):
                right_side = start
            else: 
                right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]  
            mask[i,left_side:right_side] = 1
            mask[left_side:right_side,i] = 1

    mask = mask.bool()

    return mask,all_size

class convLayer(nn.Module):
    def __init__(self,dmodel=12,window_size=2):
        super().__init__()
        self.conv = nn.Conv2d(dmodel,dmodel,[1,window_size],[1,window_size])
    def forward(self,x):
        # b,t,n,d
        b,t,n,d = x.size()
        x = x.reshape(b,d,n,t)
        out = self.conv(x) # b,d,n,t -> b,d,n,t/2
        out = F.relu(out) # b,d,n,t/2
        out = out.reshape(b,-1,n,d) # b,d,n,t/2 -> b,t/2,n,d
        return out

class convs(nn.Module):
    def __init__(self,dmodel=2,window_size=[2,2,3]):
        super().__init__()
        self.convs = nn.ModuleList([
            convLayer(dmodel,window_size[0]),
            convLayer(dmodel,window_size[1]),
            convLayer(dmodel,window_size[2])
        ])

        self.norm = nn.LayerNorm(dmodel)
    def forward(self,x):
        all_inputs = []
        all_inputs.append(x)
        out = x
        for i in range(len(self.convs)):
            out = self.convs[i](out)  # b,t/2,n,d
            all_inputs.append(out)  
        all_inputs = torch.cat(all_inputs,dim=1) # b,(t+t/2+t/4+t/12),n,d
        all_inputs = self.norm(all_inputs)
        return all_inputs

class multiHeadAttention(nn.Module):
    def __init__(self,cin=2,cout=64,nhead=8,dk=8,mask=True,all_size=[12,6,3,1],pred_num=6):
        super().__init__()
        self.nhead = nhead
        self.dk = dk 
        self.dmodel = nhead * dk
        self.fc_q = nn.Linear(cin,cout)
        self.fc_k = nn.Linear(cin,cout)
        self.fc_v = nn.Linear(cin,cout)
        self.fc_c = nn.Linear(cout,1)
        self.fc = nn.Linear(sum(all_size),pred_num)
        self.mask = True
        self.his_num=all_size[0]
    def forward(self,q,k,v,mask=None): # b,m,n,d
        batch_size = q.shape[0]

        q = self.fc_q(q) # b,m,n,d -> b,m,n,h
        k = self.fc_k(k) # b,m,n,d -> b,m,n,h
        v = self.fc_v(v) # b,m,n,d -> b,m,n,h

        # b,m,n,h -> b*nhead,m,n,dk -> b*nhead,n,m,dk 
        q = torch.cat( torch.split(q,self.dk,dim=-1),dim=0 ).permute(0,2,1,3) # b*nhead,n,m,dk
        # print("q:",q.shape)
        k = torch.cat( torch.split(k,self.dk,dim=-1),dim=0 ).permute(0,2,3,1) # b*nhead,n,dk,m
        v = torch.cat( torch.split(v,self.dk,dim=-1),dim=0 ).permute(0,2,1,3) # b*nhead,n,m,dk

        att = torch.matmul(q,k) # b*nhead,n,m,dk  @  b*nhead,n,m,dk -> b*nhead,n,m,m
        # print(att.shape)
        att /= (self.dk ** 2) 

        if self.mask:
            nums = torch.tensor(-2 ** 15).to(torch.float32)
            att = torch.where(mask,att,nums)
        
        att = F.softmax(att,dim=-1)

        out = torch.matmul(att,v)  # b*nhead,n,m,m  @  b*nhead,n,m,dk = b*nhead,n,m,dk
        out = torch.cat( torch.split(out,batch_size,dim=0),dim=-1 ) # b*nhead,n,m,dk -> b,n,m,h
        # print(out.shape)
        out = self.fc_c(out) # b,n,m,h -> b,n,m,d
        out = out.permute(0,1,3,2) # b,n,m,d -> b,n,d,m
        out = self.fc(out) # b,n,d,m -> b,n,d,pred_num
        out = out.permute(0,3,1,2).squeeze(-1)  # b,n,d,pred_num -> b,pred_num,n,d=1
        return out

if __name__ == "__main__":
    # b,t,n,d
    x = torch.randn(64,12,50,2)
    print(x.shape)

    mask, all_size = get_mask(input_size=x.shape[1], window_size=[2, 2, 3], inner_size=3)
    # print(mask.shape,all_size)

    model1 = convs(dmodel=2,window_size=[2, 2, 3])
    out1 = model1(x) # b,t,n,d -> b,(t+t/2+t/4+t/12),n,d = b,m,n,d
    # print(out1.shape)

    model2 = multiHeadAttention(cin=2,cout=64,nhead=8,dk=8)
    out2 = model2(out1,out1,out1,mask) 
    print(out2.shape)

