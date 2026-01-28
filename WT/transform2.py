import torch
import torch.nn as nn

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda() #

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        dwt_1 = dwt_init(x)
        ll1 = dwt_1[:, :channels, :, :]           # LL1 子带
        hl1 = dwt_1[:, channels:channels*2, :, :] # HL1 子带  
        lh1 = dwt_1[:, channels*2:channels*3, :, :] # LH1 子带
        hh1 = dwt_1[:, channels*3:channels*4, :, :] # HH1 子带
        dwt_2 = dwt_init(ll1)
        
        ll1[:,:,0:height//4,0:width//4] = dwt_2[:,:channels,:,:]
        ll1[:,:,height//4:height//2,0:width//4] = dwt_2[:,channels:2*channels,:,:]
        ll1[:,:,0:height//4,width//4:width//2] = dwt_2[:,2*channels:3*channels,:,:]
        ll1[:,:,height//4:height//2,width//4:width//2] = dwt_2[:,3*channels:4*channels,:,:]
        output = torch.cat([ll1, hl1, lh1, hh1], dim=1)
        
        return output


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        batch_size, total_channels, reduced_height, reduced_width = x.shape
        channels = total_channels // 4
        dwt_2 = torch.cat([x[:, :channels, 0:reduced_height//2, 0:reduced_width//2], x[:, :channels, reduced_height//2:reduced_height, 0:reduced_width//2], x[:, :channels, 0:reduced_height//2, reduced_width//2:reduced_width], x[:, :channels, reduced_height//2:reduced_height, reduced_width//2:reduced_width]], dim=1)
        
        hl1 = x[:, channels*1:channels*2, :, :]  # HL1
        lh1 = x[:, channels*2:channels*3, :, :]  # LH1  
        hh1 = x[:, channels*3:channels*4, :, :]  # HH1
        
        ll1 = iwt_init(dwt_2)
        dwt_1 = torch.cat([ll1, hl1, lh1, hh1], dim=1)
        reconstructed = iwt_init(dwt_1)
        return reconstructed


