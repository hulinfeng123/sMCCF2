# -*- coding: utf-8 -*-
import os
import torch
import random
import pickle
import numpy as np
import torch.nn as nn

data = []
for line in open('./business_user.txt', 'r'):
    (business, user, rating) = line.split(' ')
    data.append((business, user, rating))
    #   （商品i，用户u，评分r）三元组，共30838项

random.shuffle(data)

length = data.__len__()

#   train与test 8：2分成
data_train = data[:int(length * 0.8)]
data_test = data[int(length * 0.8):]

i_train, u_train, r_train = [], [], []
for i in range(len(data_train)):
    i_train.append(int(data_train[i][0]))
    u_train.append(int(data_train[i][1]))
    r_train.append(int(data_train[i][2]))
#   i_train, u_train, r_train分别记录的是训练集三元组中的对应各自编号

i_test, u_test, r_test = [], [], []
for i in range(len(data_test)):
    i_test.append(int(data_test[i][0]))
    u_test.append(int(data_test[i][1]))
    r_test.append(int(data_test[i][2]))

# data_train data_test是存着元组的列表
# i_train等是存有编码的列表。形为[0,1,2,3]

u_adj = {}  # 1286项
i_adj = {}  # 2500+。 key是编码，值是列表，列表内是多个元组，包含所有对应的用户/项目、评分。
for i in range(len(u_train)):
    if u_train[i] not in u_adj.keys():
        u_adj[u_train[i]] = []
    if i_train[i] not in i_adj.keys():
        i_adj[i_train[i]] = []
    u_adj[u_train[i]].extend([(i_train[i], r_train[i])])
    # u_adj是把一个用户对多个商品的评分组合在一行中的表示,所以需要用字典，如{"u1":"(i1,r1),(i2,r2),..."}
    i_adj[i_train[i]].extend([(u_train[i], r_train[i])])

n_users = 1286
n_items = 2614

ufeature = {}  # 1286*2614   key是用户编码，值是列表表示的对应物品位置的评分
for i in range(n_users):
    ufeature[i] = [0 for _ in range(n_items)]
#   ufeature的one-hot编码的初始化

ifeature = {}  # 2614*1286
for i in range(n_items):
    ifeature[i] = [0 for _ in range(n_users)]

for key in u_adj.keys():
    n = u_adj[key].__len__()
    for i in range(n):
        ufeature[key][u_adj[key][i][0]] = u_adj[key][i][1]
#   ufeature：将每个user编码为one-hot编码，每个user对应的2614个位置分别为user对该item的评分

for key in i_adj.keys():
    n = i_adj[key].__len__()
    for i in range(n):
        ifeature[key][i_adj[key][i][0]] = i_adj[key][i][1]

ufeature_size = ufeature[0].__len__()
#   2614
ifeature_size = ifeature[0].__len__()
#   1286

ufea = []
#   ufea存储每个user编号（0~1286）对应的one-hot编码（带有评分）
#   然后将ufea作为参数给u2e的权重赋值作为初始化权重
for key in ufeature.keys():
    ufea.append(ufeature[key])
ufea = torch.Tensor(np.array(ufea, dtype=np.float32))
u2e = nn.Embedding(n_users, ufeature_size)
u2e.weight = torch.nn.Parameter(ufea)
ifea = []
for key in ifeature.keys():
    ifea.append(ifeature[key])
ifea = torch.Tensor(np.array(ifea, dtype=np.float32))
i2e = nn.Embedding(n_items, ifeature_size)
i2e.weight = torch.nn.Parameter(ifea)

with open('./_allData.p', 'wb') as meta:
    pickle.dump((u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj), meta)

'''
ufea (tensor) 1286*2614
u2e (embedding) 1286*2614  u2e.weight 1286*2614 u2e.embedding_dim=2614
ifea  2614*1286
i2e  2614*1286  embedding_dim=1286

i_adj 2500+项的字典 一共有2614个商品 但是有的商品没有交互
u_adj 1286项的字典

i_train 24670项
i_test 6168项
u_train 24670
'''
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