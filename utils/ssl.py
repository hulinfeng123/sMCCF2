import scipy.sparse as sp
import torch.nn as nn
import numpy as np
import torch
import random

'''
ssl模块一定要使用和前面模块一样的初始化的u_embedding,i_embedding，
这样的做法都是对统一的嵌入进行优化
'''

class SSL(nn.Module):

    def __init__(self, user_embedding , item_embedding, dataset , trainUser, trainItem):

        super(SSL, self).__init__()

        self.weights = dict()

        self.dataset=dataset
        #self.embedding_size = 64
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        #self.n_layers = layer
        self.n_layers = 3
        #elf.aug_type = aug_type
        self.aug_type = 1
        self.ssl_mode = 'both_side'
        # self.ssl_ratio = ssl_ratio
        # self.ssl_temp = ssl_temp
        # self.ssl_reg = ssl_reg
        self.ssl_ratio = 0.5
        self.ssl_temp = 0.5
        self.ssl_reg = 0.5

        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.m_items

        self.training_user = trainUser
        self.training_item = trainItem

#   传入的是nodes_u  nodes_i
    def forward(self, nodes_u, nodes_i):
        #initializer = tf.contrib.layers.xavier_initializer()
        # self.weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.embedding_size]),
        #                                              name='user_embedding')
        # self.weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.embedding_size]),
        #                                              name='item_embedding')
        sub_mat = {}
        if self.aug_type in [0, 1]:
            sub_mat['adj_indices_sub1'], sub_mat['adj_values_sub1'], sub_mat[
                'adj_shape_sub1'] = self._convert_csr_to_sparse_tensor_inputs(
                self.create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
            sub_mat['adj_indices_sub2'], sub_mat['adj_values_sub2'], sub_mat[
                'adj_shape_sub2'] = self._convert_csr_to_sparse_tensor_inputs(
                self.create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
        else:
            for k in range(1, self.n_layers + 1):
                sub_mat['adj_indices_sub1%d' % k], sub_mat['adj_values_sub1%d' % k], sub_mat[
                    'adj_shape_sub1%d' % k] = self._convert_csr_to_sparse_tensor_inputs(
                    self.create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
                sub_mat['adj_indices_sub2%d' % k], sub_mat['adj_values_sub2%d' % k], sub_mat[
                    'adj_shape_sub2%d' % k] = self._convert_csr_to_sparse_tensor_inputs(
                    self.create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
        for k in range(1, self.n_layers + 1):
            if self.aug_type in [0, 1]:
                # 每一层的子矩阵都是相同的
                sub_mat['sub_mat_1%d' % k] = torch.sparse.FloatTensor(
                    sub_mat['adj_indices_sub1'],
                    sub_mat['adj_values_sub1'],
                    sub_mat['adj_shape_sub1']).to_dense().cuda()
                sub_mat['sub_mat_2%d' % k] = torch.sparse.FloatTensor(
                    sub_mat['adj_indices_sub2'],
                    sub_mat['adj_values_sub2'],
                    sub_mat['adj_shape_sub2']).to_dense().cuda()
            else:
                ''' 带有参数k代表着随机游走的每一层都是不同的图 '''
                sub_mat['sub_mat_1%d' % k] = torch.sparse.FloatTensor(
                    sub_mat['adj_indices_sub1%d' % k],
                    sub_mat['adj_values_sub1%d' % k],
                    sub_mat['adj_shape_sub1%d' % k]).to_dense().cuda()
                sub_mat['sub_mat_2%d' % k] = torch.sparse.FloatTensor(
                    sub_mat['adj_indices_sub2%d' % k],
                    sub_mat['adj_values_sub2%d' % k],
                    sub_mat['adj_shape_sub2%d' % k]).to_dense().cuda()

        #print(self.weights['user_embedding'].weight[0])
        # for i in len(range(self.weights['user_embedding'])):
        # print(self.user_embedding.size())
        # print(self.item_embedding.size())
        ego_embeddings = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        # ego_embeddings = torch.cat([self.weights['user_embedding'], self.weights['item_embedding']], dim=0)
        ego_embeddings_sub1 = ego_embeddings
        ego_embeddings_sub2 = ego_embeddings
        all_embeddings = [ego_embeddings]
        all_embeddings_sub1 = [ego_embeddings_sub1]
        all_embeddings_sub2 = [ego_embeddings_sub2]

        # print(self.user_embedding.size())
        # print(self.item_embedding.size())
        # print(ego_embeddings.size())
        # print(sub_mat['sub_mat_12'].size())
        # #   3900*3900
        # print(ego_embeddings_sub1.size())
        #   2572*2614
        for k in range(1, self.n_layers + 1):
            '''
            sparse_tensor_dense_matmul稀疏张量*稠密矩阵，返回稠密矩阵
            ego_embeddings对应E(K+1) ，E(K+1)=E(K)*归一化的A
            all_embeddings += [ego_embeddings]是迭代每层的最终结果
            后面2个是对2个不同子图对应的操作
            '''

        '''-----------------------------------------------------------------------
        这里遇到的主要问题是u2e的embedding是1286*2614的，i2e的embedding是2614*1286的，维度没法叠加
        ll师兄的tensor初始化是在模块的init中，且嵌入向量的维度大小相同，不知道改成同一纬度对别的模块有无影响       
        ----------------------------------------------------------------------------
        '''
        ego_embeddings_sub1 = torch.matmul(sub_mat['sub_mat_1%d' % k],ego_embeddings_sub1)
        all_embeddings_sub1 += [ego_embeddings_sub1]

        ego_embeddings_sub2 = torch.matmul(sub_mat['sub_mat_2%d' % k],ego_embeddings_sub2)
        all_embeddings_sub2 += [ego_embeddings_sub2]

        '''
        reduce_mean()函数沿指定轴求平均，keepdims用来设置是否维持维度，false则降维
        '''
        #all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        #u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        all_embeddings_sub1 = torch.stack(all_embeddings_sub1, dim=1)
        all_embeddings_sub1 = torch.mean(all_embeddings_sub1, dim=1, keepdims=False)
        u_g_embeddings_sub1, i_g_embeddings_sub1 = torch.split(all_embeddings_sub1, [self.n_users, self.n_items], 0)

        all_embeddings_sub2 = torch.stack(all_embeddings_sub2, dim=1)
        all_embeddings_sub2 = torch.mean(all_embeddings_sub2, dim=1, keepdims=False)
        u_g_embeddings_sub2, i_g_embeddings_sub2 = torch.split(all_embeddings_sub2, [self.n_users, self.n_items], 0)

        '''
        返回的嵌入表示分别指的是不做图结构改变的u,i嵌入表示和做了改动后生成的2个嵌入表示
        '''
        #   u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2
        if self.ssl_mode in ['user_side', 'both_side']:
            # user_emb1 = tf.nn.embedding_lookup(self.u_g_embeddings_sub1, self.users)
            # user_emb2 = tf.nn.embedding_lookup(self.u_g_embeddings_sub2, self.users)
            user_emb1 = torch.select(u_g_embeddings_sub1, nodes_u)
            user_emb2 = torch.select(u_g_embeddings_sub2, nodes_u)

            normalize_user_emb1 = torch.nn.functional.normalize(user_emb1, p=2 ,dim=1)
            normalize_user_emb2 = torch.nn.functional.normalize(user_emb2, p=2 ,dim=1)
            normalize_all_user_emb2 = torch.nn.functional.normalize(u_g_embeddings_sub2, p=2 ,dim=1)
            # pos_score_user = tf.reduce_sum(torch.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
            pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2), dim=1)
            # ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False,
            #                            transpose_b=True)
            ttl_score_user = torch.mm(normalize_user_emb1, normalize_all_user_emb2.t())

            pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
            ttl_score_user = torch(torch.exp(ttl_score_user / self.ssl_temp), dim=1)

            # ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))
            ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        if self.ssl_mode in ['item_side', 'both_side']:
            # item_emb1 = tf.nn.embedding_lookup(self.i_g_embeddings_sub1, self.pos_items)
            # item_emb2 = tf.nn.embedding_lookup(self.i_g_embeddings_sub2, self.pos_items)
            item_emb1 = torch.select(i_g_embeddings_sub1, nodes_i)
            item_emb2 = torch.select(i_g_embeddings_sub2, nodes_i)
            normalize_item_emb1 = torch.nn.functional.normalize(item_emb1, p=2 ,dim=1)
            normalize_item_emb2 = torch.nn.functional.normalize(item_emb2, p=2 ,dim=1)
            normalize_all_item_emb2 = torch.nn.functional.normalize(i_g_embeddings_sub2, p=2 ,dim=1)
            # pos_score_item = torch.sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
            pos_score_item = torch.sum(torch.mul(normalize_item_emb1, normalize_item_emb2),dim=1)
            ttl_score_item = torch.mm(normalize_item_emb1, normalize_all_item_emb2.t())

            pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
            ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)

            ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))

        if self.ssl_mode == 'user_side':
            ssl_loss = self.ssl_reg * ssl_loss_user
        elif self.ssl_mode == 'item_side':
            ssl_loss = self.ssl_reg * ssl_loss_item
        else:
            ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)

        return ssl_loss


    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.vstack((coo.row, coo.col))
        # indices = np.mat([coo.row, coo.col]).transpose()
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(coo.data)
        shape = coo.shape
        #torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
        #return indices, coo.data, coo.shape
        return i, v, shape

    def create_adj_mat(self, is_subgraph=False, aug_type=0):
        # @timer
        # aug_type参数指的是数据加强类型
        n_nodes = self.n_users + self.n_items
        if is_subgraph and aug_type in [0, 1, 2] and self.ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk

            '''
            边丢失和随机游走采用的都是对边掩码的方式，所以0与1，2的方式不同
            '''
            if aug_type == 0:
                """
                def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
                    ’Return random integers from `0` (inclusive) to `high` (exclusive).‘
                """
                drop_user_idx = random.sample(list(range(self.n_users)), size=self.n_users * self.ssl_ratio)
                drop_item_idx = random.sample(list(range(self.n_items)), size=self.n_items * self.ssl_ratio)
                indicator_user = np.ones(self.n_users, dtype=np.float32)
                indicator_item = np.ones(self.n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                # sp.diags是对矩阵做对角化
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                
                #######################################################
                # np.ones_like(self.training_user, dtype=np.float32)这一句需要改为具体的用户对item的评分
                
                R = sp.csr_matrix(
                    (np.ones_like(self.training_user, dtype=np.float32), (self.training_user, self.training_item)),
                    shape=(self.n_users, self.n_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data

                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.n_users)),
                                        shape=(n_nodes, n_nodes))
                ''' 建立U-I的节点丢失后的二部邻接图'''

            if aug_type in [1, 2]:
                keep_idx = random.sample(list(range(len(self.training_user))),int(len(self.training_user) * (1 - self.ssl_ratio)))
                user_np = np.array(self.training_user)[keep_idx]
                item_np = np.array(self.training_item)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)), shape=(n_nodes, n_nodes))
        else:
            user_np = np.array(self.training_user)
            item_np = np.array(self.training_item)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        '''
         flatten()函数用于降成1维
         这里做的操作就是lightgcn里的A归一化操作
        '''
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # print('use the pre adjcency matrix')

        return adj_matrix
