# -*- coding: utf-8 -*-
import os
import torch
import random
import pickle
import numpy as np
import torch.nn as nn

# with open('./_allData.p', 'rb') as meta:
#     u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj = pickle.load(meta)
# u_neighs = {}
# def check(l1, l2):
#     x = 10  # 要求x件以上相同商品id和评分都相同
#     cnt = 0
#     for item1 in l1:
#         for item2 in l2:
#             if item1[0] == item2[0] and item1[1] == item2[1]:
#                 cnt += 1
#     if cnt >= x:
#         return True
#     else:
#         return False
#
# u_neigh = {}
# ids = list(u_adj.keys())
# for id in ids:
#     if id not in u_neigh.keys():
#         u_neigh[id] = []
# length = ids.__len__()
#
# for i in range(0, length):
#     for j in range(i + 1, length):
#         if check(u_adj[i], u_adj[j]):
#             u_neigh[i].append(j)
#             u_neigh[j].append(i)
#             # print(str(ids[i]) + " " + str(ids[j]) + " is matched")
# print("Finished!")
#
# for id in ids:
#     print(u_neigh[id])
with open('./_allData.p', 'rb') as meta:
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

with open('./friends.p', 'wb') as meta2:
    pickle.dump((u_friends, i_friends), meta2)

# with open('./friends.p', 'rb') as meta2:
#     u_neighs, i_neighs = pickle.load(meta2)
# for i in range(0, 10):
#     print(u_neighs[i], '\n')
# for i in range(0, 10):
#     print(i_neighs[i], '\n')
#
