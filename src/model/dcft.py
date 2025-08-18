import torch.nn as nn
import torch
class DCFT(nn.Module):

    def __init__(self, layer, d, k, dropout_rate):
        super().__init__()
        self.base_layer = nn.Parameter(layer.weight.data.clone(), requires_grad=False)
        self.d_in = layer.in_features
        self.d_out = layer.out_features
        self.d = d
        self.k = k
        self.orthogonal_loss = 0.0
        self.B = nn.Parameter(torch.randn((self.d_in // self.d, self.k)))
        self.A = nn.Parameter(torch.randn((self.k, self.d_out // self.d)))
        self.deconv = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=d,
            stride=d,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def get_orthogonal_loss(self):
        return self.orthogonal_loss

    def update_orthogonal_loss(self):
        a_orth = self.A.T @ self.A - torch.eye(self.d_out // self.d, device=self.A.device)
        b_orth = self.B @ self.B.T - torch.eye(self.d_in // self.d, device=self.B.device)
        self.orthogonal_loss = torch.norm(a_orth, 'fro')**2 + torch.norm(b_orth, 'fro') ** 2

    def forward(self, x):
        F = self.B @ self.A
        F = F.view(1, 1, self.d_in//self.d, self.d_out//self.d)
        delta = self.deconv(F)
        delta = delta.view(self.d_out, self.d_in)
        layer = self.base_layer + delta
        out = torch.einsum('bsi,ei->bse', x, layer)
        out = self.dropout(out)
        self.update_orthogonal_loss()
        return out


