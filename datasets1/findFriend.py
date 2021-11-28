# -*- coding: utf-8 -*-
import os
import torch
import random
import pickle
import numpy as np
import torch.nn as nn

with open('./yelp/_allData.p', 'rb') as meta:
    u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj = pickle.load(meta)
u_friends = {}  # 1286
i_friends = {}  # 2535

# user part
for user in set(u_train):
    if user not in u_friends.keys():
        u_sim = {}  #  存该user的各个邻居的sim值
        for friend in set(u_train):
            if friend not in u_sim.keys() and user != friend:
                sim = torch.cosine_similarity(u2e.weight[user], u2e.weight[friend], dim=0)  # 余弦相似度
                if sim > 0: u_sim[friend] = sim
        ufriends = []
        if len(u_sim) > 24:
            ufriends = [x[0] for x in sorted(u_sim.items(), key=lambda y: y[1], reverse=True)[:24]]
        else:
            ufriends = list(u_sim.keys())
        u_friends[user] = ufriends
print(len(u_friends))
# for i in range(0, 10):
#     print(u_friends[i], '\n')
# item part
for item in set(i_train):
    if item not in i_friends.keys():
        i_sim = {}
        for friend in set(i_train):
            if friend not in i_sim.keys() and item != friend:
                sim = torch.cosine_similarity(i2e.weight[item], i2e.weight[friend], dim=0)
                if sim > 0: i_sim[friend] = sim
        ifriends = []
        if len(i_sim) > 12:
            ifriends = [x[0] for x in sorted(i_sim.items(), key=lambda y: y[1], reverse=True)[:24]]
        else:
            ifriends = list(i_sim.keys())
        i_friends[item] = ifriends
print(len(i_friends))
# for i in range(0, 10):
#     print(i_friends[i], '\n')

with open('./yelp/friends.p', 'wb') as meta2:
    pickle.dump((u_friends, i_friends), meta2)

# with open('./friends.p', 'rb') as meta2:
#     u_neighs, i_neighs = pickle.load(meta2)
# for i in range(0, 10):
#     print(u_neighs[i], '\n')
# for i in range(0, 10):
#     print(i_neighs[i], '\n')
#
