import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class combiner(nn.Module):
    def __init__(self, embedding1, memory, embedding_dim, droprate, device='cpu', is_user_part=True):
        super(combiner, self).__init__()

        self.embedding1 = embedding1
        self.memory = memory
        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = device
        self.is_user = is_user_part
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 2)
        # self.att2 = nn.Linear(self.embed_dim, 1)
        # self.softmax = nn.Softmax()

    def forward(self, nodes_u, nodes_i):
        embedding1 = self.embedding1(nodes_u, nodes_i)   # 256*embed_dim
        embed_matrix_f = self.memory(nodes_u, nodes_i, self.is_user)
        x1 = torch.cat((embedding1, embed_matrix_f), dim=1)
        x1 = F.relu(self.att1(x1).to(self.device), inplace=True)
        x1 = F.dropout(x1, training=self.training, p=self.droprate)
        x1 = F.relu(self.att2(x1).to(self.device), inplace=True)
        x1 = F.dropout(x1, training=self.training, p=self.droprate)
        att = F.softmax(x1, dim=1)
        att1, att2 = att.chunk(2, dim=1)
        final_embed_matrix = torch.mul(embedding1, att1) + torch.mul(embed_matrix_f, att2)
        # x1 = F.relu(self.att1(embedding1).to(self.device), inplace=True)  # 公式（8）
        # x1 = F.dropout(x1, training=self.training)
        # x2 = F.relu(self.att1(embed_matrix_f).to(self.device), inplace=True)
        # x2 = F.dropout(x2, training=self.training)
        # x = self.att2(x).to(self.device)
        # att_w = F.softmax(x, dim=1)  # 256*3
        # print(att_w1.shape)
        # print(att_w1.shape)
        # final_embed_matrix = torch.mul(embedding1, att_w)  # 256*32 * 256*1
        # print(final_embed_matrix.shape)
        return final_embed_matrix  # 公式（10）  # 256*32
