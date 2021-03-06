import torch
import heapq
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.l0dense import L0Dense
from utils.attention import attention


class aggregator(nn.Module):
    """
    aggregator: for aggregating feature of neighbors

    """

    ##############################
    #   u2e和i2e不能轻易删除，里面是包含了带有评分的初始化向量
    def __init__(self, u_feature, i_feature, adj, embed_dim, weight_decay=0.0005, droprate=0.5,
                 device='cpu', is_user_part=True):
        super(aggregator, self).__init__()
        self.ufeature = u_feature
        self.ifeature = i_feature
        self.adj = adj
        self.embed_dim = embed_dim
        self.weight_decay = weight_decay
        self.droprate = droprate
        self.device = device
        self.is_user = is_user_part
        #   u_layer是2614*256的tensor
        #   i_layer是1286*256的tensor
        self.u_layer = L0Dense(self.ufeature.embedding_dim, self.embed_dim,
                               weight_decay=self.weight_decay, droprate=self.droprate, device=self.device)
        self.i_layer = L0Dense(self.ifeature.embedding_dim, self.embed_dim,
                               weight_decay=self.weight_decay, droprate=self.droprate, device=self.device)
        self.att = attention(self.embed_dim, self.droprate, device=self.device)

    def forward(self, nodes):
        if self.is_user:
            embed_matrix = torch.empty(self.ufeature.num_embeddings, self.embed_dim,
                                       dtype=torch.float).to(self.device)
            nodes_fea = self.u_layer(self.ufeature.weight[nodes]).to(self.device)
            threshold = 24  # 阈值
        else:
            embed_matrix = torch.empty(self.ifeature.num_embeddings, self.embed_dim,
                                       dtype=torch.float).to(self.device)
            nodes_fea = self.i_layer(self.ifeature.weight[nodes]).to(self.device)
            threshold = 12

        length = list(nodes.size())
        for i in range(length[0]):
            index = nodes[[i]].cpu().numpy()
            if self.training:
                interactions = self.adj[index[0]]
            else:
                if index[0] in self.adj.keys():
                    interactions = self.adj[index[0]]
                else:
                    if self.is_user:
                        node_feature = self.u_layer(self.ufeature.weight[torch.LongTensor(index)]).to(self.device)
                    else:
                        node_feature = self.i_layer(self.ifeature.weight[torch.LongTensor(index)]).to(self.device)

                    embed_matrix[index] = node_feature
                    continue

            n = len(interactions)
            if n < threshold:
                interactions = [item[0] for item in interactions]
                times = n
            else:
                interactions = a_res(interactions, threshold)
                times = threshold

            if self.is_user:  # 公式（1）
                # use part
                neighs_feature = self.i_layer(self.ifeature.weight[torch.LongTensor(interactions)]).to(self.device)
                node_feature = self.u_layer(self.ufeature.weight[torch.LongTensor(index)]).to(self.device)
            else:  # 公式（2）
                # item part
                neighs_feature = self.u_layer(self.ufeature.weight[torch.LongTensor(interactions)]).to(self.device)
                node_feature = self.i_layer(self.ifeature.weight[torch.LongTensor(index)]).to(self.device)

            att_w = self.att(neighs_feature, node_feature, times)
            embedding = torch.mm(neighs_feature.t(), att_w)  # 32*time * time*1 -> 32*1
            embed_matrix[index] = embedding.t()
        return nodes_fea, embed_matrix


def a_res(samples, m):
    """

    :param samples: [(entity, weight), ...]
    :param m: number of selected entities
    :return: [(entity, weight), ...]
    """
    heap = []
    for sample in samples:
        w = sample[1]
        u = random.uniform(0, 1)
        k = u ** (1 / w)

        if len(heap) < m:
            heapq.heappush(heap, (k, sample))
        elif k > heap[0][0]:
            heapq.heappush(heap, (k, sample))

            if len(heap) > m:
                heapq.heappop(heap)
    return [item[1][0] for item in heap]
