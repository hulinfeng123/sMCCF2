import torch
from torch import nn


class test(nn.Module):
    def __init__(self, embed_dim=32, N=8):
        super(test, self).__init__()
        self.N = N
        self.embed_dim = embed_dim
        self.M = [nn.Parameter(torch.zeros(size=(1, embed_dim))) for _ in range(N)]
        self.K = [nn.Linear(embed_dim, 1, bias=False) for _ in range(N)]
        for i in range(N):
            nn.init.normal_(self.M[i].data, std=0.1)

    def forward(self, u_i, u_il):
        s = torch.mul(u_i, u_il)
        s = s / u_i.norm(2) / u_il.norm(2)
        alpha = torch.randn([1, self.N])
        for i in range(len(self.K)):
            alpha[0][i] = self.K[i](s)
        alpha = torch.softmax(alpha, dim=0)

        V = torch.randn([self.N, self.rec_dim])
        for i in range(len(self.M)):
            V[i] = torch.mul(u_il, self.M[i].data)

        f = torch.mm(alpha, V)
