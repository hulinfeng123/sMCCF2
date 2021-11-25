import scipy.sparse as sp
import torch.nn as nn
import numpy as np
import torch



class SSL(nn.Module):

    def __init__(self, U2E , I2E, dataset , embed_dim ,layer , aug_type , ssl_reg , ssl_temp , ssl_ratio):

        super(SSL, self).__init__()

        self.dataset=dataset
        self.embedding_size = 64
        self.user_embedding = U2E
        self.item_embedding = I2E
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

        self.n_users = self.dataset.n_user
        self.n_items = self.dataset.m_item

        self.training_user = self.dataset.trainUser
        self.training_item = self.dataset.trainItem

#   传入的是nodes_u  nodes_i
    def forward(self, nodes_u, nodes_i):
        self.weights = dict()
        self.weights['user_embedding'] = self.user_embedding
        self.weights['item_embedding'] = self.item_embedding
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
                self.sub_mat['sub_mat_1%d' % k] = torch.sparse.Tensor(
                    self.sub_mat['adj_indices_sub1'],
                    self.sub_mat['adj_values_sub1'],
                    self.sub_mat['adj_shape_sub1'])
                self.sub_mat['sub_mat_2%d' % k] = torch.sparse.Tensor(
                    self.sub_mat['adj_indices_sub2'],
                    self.sub_mat['adj_values_sub2'],
                    self.sub_mat['adj_shape_sub2'])
            else:
                ''' 带有参数k代表着随机游走的每一层都是不同的图 '''
                self.sub_mat['sub_mat_1%d' % k] = torch.sparse.Tensor(
                    self.sub_mat['adj_indices_sub1%d' % k],
                    self.sub_mat['adj_values_sub1%d' % k],
                    self.sub_mat['adj_shape_sub1%d' % k])
                self.sub_mat['sub_mat_2%d' % k] = torch.sparse.Tensor(
                    self.sub_mat['adj_indices_sub2%d' % k],
                    self.sub_mat['adj_values_sub2%d' % k],
                    self.sub_mat['adj_shape_sub2%d' % k])
        ego_embeddings = torch.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        ego_embeddings_sub1 = ego_embeddings
        ego_embeddings_sub2 = ego_embeddings
        all_embeddings = [ego_embeddings]
        all_embeddings_sub1 = [ego_embeddings_sub1]
        all_embeddings_sub2 = [ego_embeddings_sub2]

        for k in range(1, self.n_layers + 1):
            '''
            sparse_tensor_dense_matmul稀疏张量*稠密矩阵，返回稠密矩阵
            ego_embeddings对应E(K+1) ，E(K+1)=E(K)*归一化的A
            all_embeddings += [ego_embeddings]是迭代每层的最终结果
            后面2个是对2个不同子图对应的操作
            '''
            #ego_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")
            #all_embeddings += [ego_embeddings]

            # ego_embeddings_sub1 = tf.sparse_tensor_dense_matmul(
            #     self.sub_mat['sub_mat_1%d' % k],
            #     ego_embeddings_sub1, name="sparse_dense_sub1%d" % k)
            # ego_embeddings_sub1 = tf.multiply(ego_embeddings_sub1, self.mask1)
            ego_embeddings_sub1 = torch.matmul(
                self.sub_mat['sub_mat_1%d' % k],
                ego_embeddings_sub1, name="sparse_dense_sub1%d" % k)
            all_embeddings_sub1 += [ego_embeddings_sub1]

            ego_embeddings_sub2 = torch.matmul(
                self.sub_mat['sub_mat_2%d' % k],
                ego_embeddings_sub2, name="sparse_dense_sub2%d" % k)
            # ego_embeddings_sub2 = tf.multiply(ego_embeddings_sub2, self.mask2)
            all_embeddings_sub2 += [ego_embeddings_sub2]

        #all_embeddings = tf.stack(all_embeddings, 1)
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

    def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
        """Return random integers from `0` (inclusive) to `high` (exclusive).
        """
        a = np.arange(high)
        if exclusion is not None:
            if p is None:
                p = np.ones_like(a)
            else:
                p = np.array(p, copy=True)
            p = p.flatten()
            p[exclusion] = 0
        if p is not None:
            p = p / np.sum(p)
        sample = np.random.choice(a, size=size, replace=replace, p=p)
        return sample

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

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
                drop_user_idx = self.randint_choice(self.n_users, size=self.n_users * self.ssl_ratio, replace=False)
                drop_item_idx = self.randint_choice(self.n_items, size=self.n_items * self.ssl_ratio, replace=False)
                indicator_user = np.ones(self.n_users, dtype=np.float32)
                indicator_item = np.ones(self.n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                # sp.diags是对矩阵做对角化
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
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
                keep_idx = ssl_tool.randint_choice_v2(len(self.training_user),
                                             size=int(len(self.training_user) * (1 - self.ssl_ratio)), replace=False)
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
