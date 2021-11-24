import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, embedding_dim, aggregator, device='cpu', is_user_part=True):
        super(encoder, self).__init__()

        self.embed_dim = embedding_dim
        self.aggregator = aggregator
        self.device = device
        self.is_user = is_user_part
        self.layer = nn.Linear(self.embed_dim * 2, self.embed_dim)

    def forward(self, nodes_u, nodes_i):  # 公式（7）
        # self-connection could be considered
        if self.is_user:
            nodes_fea, embed_matrix = self.aggregator(nodes_u)
            # print("nodes_u: ", nodes_u, nodes_u.shape)
            # print("nodes_u.cpu().numpy: ", nodes_u.cpu().numpy(), nodes_u.shape)
            # print("nodes_fea: ", nodes_fea, nodes_fea.shape) # 256*32
            # print("embed_matrix: ", embed_matrix, embed_matrix.shape)  # 1286*32
            combined = torch.cat((nodes_fea, embed_matrix[nodes_u.cpu().numpy()]), dim=1)
            # print("combined: ", combined, combined.shape)
        else:
            nodes_fea, embed_matrix = self.aggregator(nodes_i)
            combined = torch.cat((nodes_fea, embed_matrix[nodes_i.cpu().numpy()]), dim=1)

        cmp_embed_matrix = self.layer(combined).to(self.device)
        # print("cmp_embed_matrix: ", cmp_embed_matrix, cmp_embed_matrix.shape) # 256*32

        return cmp_embed_matrix
