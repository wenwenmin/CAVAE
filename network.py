import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#MultiHeadAttention
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)
    def forward(self, q, k, v, mask=None):
        u = torch.matmul(q, k.transpose(-2, -1))
        u = u / self.scale
        if mask is not None:
            u = u.masked_fill(mask, -np.inf)
        attn = self.softmax(u)
        output = torch.matmul(attn, v)
        return attn, output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)
        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
        self.fc_o = nn.Linear(n_head * d_v, d_o)
    def forward(self, q, k, v, mask=None):
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v
        n_q, d_q_ = q.size()
        n_k, d_k_ = k.size()
        n_v, d_v_ = v.size()
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(n_q, n_head, d_q).permute(1, 0, 2).contiguous().view(-1, n_q, d_q)
        k = k.view(n_k, n_head, d_k).permute(1, 0, 2).contiguous().view(-1, n_k, d_k)
        v = v.view(n_v, n_head, d_v).permute(1, 0, 2).contiguous().view(-1, n_v, d_v)
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, n_q, d_v).permute(0, 1, 2).contiguous().view(n_q, -1)
        output = self.fc_o(output)
        return attn, output

class MultiSelfAttention(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))
        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)
        self.init_parameters()
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)
    def forward(self, x, mask=None):
        x = x.to(device)
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)
        attn, output = self.mha(q, k, v, mask=mask)
        return output

class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()
    def forward(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.shape)
        eps = eps.to(device)
        std = torch.exp(z_log_var / 2)
        out = z_mean + std*eps
        return out

class Encoder(nn.Module):
    def __init__(self, dropout, bn):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(400, 256),
            nn.Dropout(p=dropout, inplace=False),
            nn.BatchNorm1d(256, eps=bn),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=dropout, inplace=False),
            nn.BatchNorm1d(128, eps=bn),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=dropout, inplace=False),
            nn.BatchNorm1d(64, eps=bn),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(64, 32)
        self.z_log_var = nn.Linear(64, 32)
        self.sample = Sample()
        self.classfier = nn.Sequential(
            nn.Linear(32,2),
            # nn.Dropout(p=dropout, inplace=False),
            # nn.BatchNorm1d(16, eps=bn),
            # nn.ReLU(),
            # nn.Linear(16, 8),
            # nn.Dropout(p=dropout, inplace=False),
            # nn.BatchNorm1d(8, eps=bn),
            # nn.ReLU(),
            # nn.Linear(8, 4),
            # nn.Dropout(p=dropout, inplace=False),
            # nn.BatchNorm1d(4, eps=bn),
            # nn.ReLU(),
            # nn.Linear(4, 2),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.model(x)
        z_mean = self.z_mean(out)
        z_log_var = self.z_log_var(out)
        out = self.sample(z_mean, z_log_var)
        out = self.classfier(out)
        return out

class MAVC(nn.Module):
    def __init__(self, n_head_, d_k_, d_v_, drop_out, batch_norml):
        super(MAVC, self).__init__()
        self.encoder = Encoder(dropout=drop_out, bn=batch_norml)
        self.attn1 = MultiSelfAttention(n_head=n_head_, d_k=d_k_, d_v=d_v_, d_x=22, d_o=100)
        self.attn2 = MultiSelfAttention(n_head=n_head_, d_k=d_k_, d_v=d_v_, d_x=300, d_o=100)

    def forward(self, x):
        x.to(device)
        cli_input = x[:, 0:22]
        cna_input = x[:, 22:322]
        rna_input = x[:, 322:622]
        mic_input = x[:, 622:922]
        cli_out = self.attn1(cli_input)
        cna_out = self.attn2(cna_input)
        mic_out = self.attn2(mic_input)
        rna_out = self.attn2(rna_input)
        x = torch.cat((cli_out, cna_out, mic_out, rna_out), dim = 1)
        x = self.encoder(x)
        return x

# dropout = 0.895
# BN = 1e-4
# num_head = 64
# K = 64
# V = 64
# model = MAVC(n_head_=num_head, d_k_=K, d_v_=V,drop_out = dropout, batch_norml = BN)
#
# print(model)
