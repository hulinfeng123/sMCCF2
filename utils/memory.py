import torch
import heapq
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.attention import attention


class memory(nn.Module):
    def __init__(self, u_feature, i_feature, friends, embed_dim, weight_decay=0.0005, droprate=0.5,
                 device='cpu', is_user_part=True):
        super(memory, self).__init__()

        self.ufeature = u_feature
        self.ifeature = i_feature
        self.friends = friends
        self.embed_dim = embed_dim
        self.droprate = droprate
        self.device = device
        self.is_user = is_user_part
        self.u_layer = nn.Linear(self.ufeature.embedding_dim, self.embed_dim)
        self.i_layer = nn.Linear(self.ifeature.embedding_dim, self.embed_dim)
        self.layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer2 = nn.Linear(self.embed_dim, self.embed_dim * 4)
        self.K = nn.Parameter(torch.randn(size=(embed_dim, 4)))
        self.att = attention(self.embed_dim, self.droprate, device=self.device)
        self.uf_layer = nn.Linear(self.ufeature.embedding_dim, self.embed_dim)
        self.if_layer = nn.Linear(self.ifeature.embedding_dim, self.embed_dim)

    def forward(self, nodes_u, nodes_i, is_user):
        self.is_user = is_user
        if self.is_user:
            nodes = nodes_u
            embed_matrix_f = torch.empty(self.ufeature.num_embeddings, self.embed_dim,
                                         dtype=torch.float).to(self.device)
            nodes_fea = self.u_layer(self.ufeature.weight[nodes]).to(self.device)
        else:
            nodes = nodes_i
            embed_matrix_f = torch.empty(self.ifeature.num_embeddings, self.embed_dim,
                                         dtype=torch.float).to(self.device)
            nodes_fea = self.i_layer(self.ifeature.weight[nodes]).to(self.device)
        length = list(nodes.size())
        for i in range(length[0]):
            index = nodes[[i]].cpu().numpy()
            if self.training:
                friends = self.friends[index[0]]
            else:
                if index[0] in self.friends.keys():
                    friends = self.friends[index[0]]
                else:
                    if self.is_user:
                        node_feature = self.u_layer(self.ufeature.weight[torch.LongTensor(index)]).to(self.device)
                    else:
                        node_feature = self.i_layer(self.ifeature.weight[torch.LongTensor(index)]).to(self.device)
                    embed_matrix_f[index] = node_feature
                    continue
            if self.is_user:
                #  user part
                node_feature = self.u_layer(self.ufeature.weight[torch.LongTensor(index)]).to(self.device)
                friends_feature = self.uf_layer(self.ufeature.weight[torch.LongTensor(friends)]).to(self.device)
                cross_friends = torch.mul(node_feature.repeat(len(friends), 1), friends_feature)
            else:
                node_feature = self.i_layer(self.ifeature.weight[torch.LongTensor(index)]).to(self.device)
                friends_feature = self.if_layer(self.ifeature.weight[torch.LongTensor(friends)]).to(self.device)
                cross_friends = torch.mul(node_feature.repeat(len(friends), 1), friends_feature)  # 朋友数*dim

            V = self.layer2(friends_feature).to(self.device)  # 朋友数*dim*4
            V = F.relu(V, inplace=True)
            V = F.dropout(V, training=self.training)
            V1, V2, V3, V4 = V.chunk(4, dim=1)
            # V1, V2, V3, V4, V5, V6 = V.chunk(6, dim=1)  # 朋友数*dim
            att_j = torch.mm(cross_friends, self.K)
            att_j = F.softmax(att_j, dim=1)
            att_j1, att_j2, att_j3, att_j4 = att_j.chunk(4, dim=1)
            # att_j1, att_j2, att_j3, att_j4, att_j5, att_j6 = att_j.chunk(6, dim=1)  # 朋友数*1
            fil = torch.mul(V1, att_j1) + torch.mul(V2, att_j2) \
                  + torch.mul(V3, att_j3) + torch.mul(V4, att_j4)
            # torch.mul(V5, att_j5) + torch.mul(V6, att_j6)   # 朋友数*dim
            att_b = self.att(fil, node_feature, len(friends)).to(self.device)  # 朋友数*1
            embedding = torch.mm(fil.t(), att_b)  # dim*朋友数 * 朋友数*1
            embed_matrix_f[index] = embedding.t()

        return embed_matrix_f[nodes]
