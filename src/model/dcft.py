import torch.nn as nn
import torch
from copy import deepcopy
class DCFT(nn.Module):

    def __init__(self, layer, d, k):
        super().__init__()
        self.base_layer = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.d_in = layer.in_features
        self.d_out = layer.out_features
        self.d = d
        self.k = k
        self.B = nn.Parameter(torch.randn((self.d_in // self.d, self.k)))
        self.A = nn.Parameter(torch.randn((self.k, self.d_out // self.d)))
        self.deconv = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=d,
            stride=d,
        )

    def forward(self, x):
        #print(x.shape) # torch.Size([32, 18, 768])
        f = self.B @ self.A
        #print(self.A.shape, self.B.shape, f.shape)  # torch.Size([1, 288]) torch.Size([96, 1]) torch.Size([96, 288])
        f = f.unsqueeze(0)
        delta = self.deconv(f)
        #print(delta.shape) # torch.Size([1, 768, 2304])
        delta = delta.squeeze(0)
        #print(delta.shape) # torch.Size([768, 2304])
        #print(self.base_layer.shape) # torch.Size([2304, 768])
        layer = self.base_layer + delta.T
        #print(layer.shape) # torch.Size([2304, 768])

        out = torch.einsum('bsi,ei->bse', x, layer)
        return out


