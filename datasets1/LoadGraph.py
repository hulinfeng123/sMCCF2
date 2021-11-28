from time import time
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
import random
import pickle
import numpy as np
import torch.nn as nn


class Loader:
    def __init__(self, dataset, device):
        super(Loader, self).__init__()

        self.n_user = 0
        self.m_item = 0
        self.device = device
        self.dataset = dataset
        train_file = "./" + dataset + "/train.txt"
        test_file = "./" + dataset + "/test.txt"
        uu_file = "./" + dataset + "/uu.txt"
        ii_file = "./" + dataset + "/ii.txt"
        self.path = "./" + dataset
        trainUniqueUsers, self.trainItem, self.trainUser, self.trainRating = [], [], [], []
        testUniqueUsers, self.testItem, self.testUser, self.testRating = [], [], [], []
        UUUniqueUsers, UUUser, UUInter = [], [], []
        IIUniqueUsers, IIItem, IIInter = [], [], []
        self.u_adj = {}
        self.i_adj = {}
        self.traindataSize = 0
        self.testdataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    item = int(l[1])
                    rating = int(l[2])

                    if uid not in trainUniqueUsers:
                        trainUniqueUsers.append(uid)
                    self.trainUser.append(uid)
                    self.trainItem.append(item)
                    self.trainRating.append(rating)

                    self.m_item = max(self.m_item, item)
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += 1

        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(self.trainUser)
        self.trainItem = np.array(self.trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    item = int(l[1])
                    rating = int(l[2])

                    if uid not in testUniqueUsers:
                        testUniqueUsers.append(uid)

                    self.testUser.append(uid)
                    self.testItem.append(item)
                    self.testRating.append(rating)

                    self.m_item = max(self.m_item, item)
                    self.n_user = max(self.n_user, uid)
                    self.testdataSize += 1

        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(self.testUser)
        self.testItem = np.array(self.testItem)

        with open(uu_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if self.dataset == 'amazon':
                        inter = [int(i) for i in l[1:] if i != '' and int(i) < 1000]
                    else:
                        inter = [int(i) for i in l[1:] if i != '']
                    uid = int(l[0])
                    UUUniqueUsers.append(uid)
                    UUUser.extend([uid] * len(inter))
                    UUInter.extend(inter)

        self.UUUniqueUsers = np.array(UUUniqueUsers)
        self.UUUser = np.array(UUUser)
        self.UUInter = np.array(UUInter)

        with open(ii_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if self.dataset == 'amazon':
                        inter = [int(i) for i in l[1:] if i != '' and int(i) < 1000]
                    else:
                        inter = [int(i) for i in l[1:] if i != '']
                    item = int(l[0])
                    IIUniqueUsers.append(item)
                    IIItem.extend([item] * len(inter))
                    IIInter.extend(inter)

        self.IIUniqueUsers = np.array(IIUniqueUsers)
        self.IIItem = np.array(IIItem)
        self.IIInter = np.array(IIInter)

        self.Graph = None
        self.UUGraph = None
        self.IIGraph = None

        print(f"{self.trainDataSize} interactions for training")
        print(f"{dataset} Sparsity : {self.trainDataSize / self.n_users / self.m_items}")

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        self.UserUserNet = csr_matrix((np.ones(len(self.UUUser)), (self.UUUser, self.UUInter)),
                                      shape=(self.n_user, self.n_user))
        self.ItemItemNet = csr_matrix((np.ones(len(self.IIItem)), (self.IIItem, self.IIInter)),
                                      shape=(self.m_items, self.m_items))

        # 统计用户交互个数
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1

        # 统计物品交互个数
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        '''--------------------------------加入u_adj以及i_adj--------------------------------------'''
        u_adj = {}
        i_adj = {}
        for i in range(len(self.trainUser)):
            if self.trainUser[i] not in u_adj.keys():
                u_adj[self.trainUser[i]] = []
            if self.trainItem[i] not in i_adj.keys():
                i_adj[self.trainItem[i]] = []
            u_adj[self.trainUser[i]].extend([(self.trainItem[i], self.trainRating[i])])
            # u_adj是把一个用户对多个商品的评分组合在一行中的表示,所以需要用字典，如{"u1":"(i1,r1),(i2,r2),..."}
            i_adj[self.trainItem[i]].extend([(self.trainUser[i], self.trainRating[i])])
        self.u_adj = u_adj
        self.i_adj = i_adj
        '''--------------------------------加入u_adj以及i_adj--------------------------------------'''
    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDataSize(self):
        return self.testdataSize

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    def getSparseGraph(self):
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                # print("successfully loaded ui adjacency matrix")
                norm_adj = pre_adj_mat
            except:
                # print("generating ui adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                # print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)
        return self.Graph

    def getUserUserGraph(self):
        if self.UUGraph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/u_u_s_pre_adj_mat.npz')
                # print("successfully loaded uu adjacency matrix")
                norm_adj = pre_adj_mat
            except:
                # print("generating uu adjacency matrix")
                s = time()
                R = self.UserUserNet.tolil()
                adj_mat = R
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                # print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/u_u_s_pre_adj_mat.npz', norm_adj)

            self.UUGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.UUGraph = self.UUGraph.coalesce().to(self.device)
        return self.UUGraph

    def getItemItemGraph(self):
        if self.IIGraph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/i_i_s_pre_adj_mat.npz')
                # print("successfully loaded ii adjacency matrix")
                norm_adj = pre_adj_mat
            except:
                # print("generating ii adjacency matrix")
                s = time()
                R = self.ItemItemNet.tolil()
                adj_mat = R
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                # print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/i_i_s_pre_adj_mat.npz', norm_adj)

            self.IIGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.IIGraph = self.IIGraph.coalesce().to(self.device)
        return self.IIGraph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

