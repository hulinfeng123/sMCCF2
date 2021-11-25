# -*- coding: utf-8 -*-
import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from utils.aggregator import aggregator
from utils.encoder import encoder
from utils.combiner import combiner
from utils.l0dense import L0Dense
from utils.memory import memory
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from utils.ssl import SSL
from utils.LoadGraph import Loader


class SMCCF(nn.Module):
    '''
    SMCCF的初始化变量u_embedding，i_embedding是经过下面3个组件生成表示后融合完成
    所以此类里面只需要对他们进行MLP
    '''
    def __init__(self, u_embedding, i_embedding, ssl_loss, dataset, embed_dim, N=30000, droprate=0.5, beta_ema=0.999, device='cpu'):
        super(SMCCF, self).__init__()

        self.u_embed = u_embedding
        self.i_embed = i_embedding
        self.ssl_loss = ssl_loss
        self.dataset = dataset
        self.embed_dim = embed_dim
        self.N = N
        self.droprate = droprate
        self.beta_ema = beta_ema
        self.device = device

        self.u_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.u_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.i_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.i_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.ui_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.ui_layer2 = nn.Linear(self.embed_dim, 1)

        self.u_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.i_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.ui_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)

        if beta_ema > 0.:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            self.avg_param = [a.to(self.device) for a in self.avg_param]
            self.steps_ema = 0.

        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_i):
        nodes_u_embed = self.u_embed(nodes_u, nodes_i)
        nodes_i_embed = self.i_embed(nodes_u, nodes_i)
        #print(self.u_embed)

        # nodes_u_embed = self.u_embed
        # nodes_i_embed = self.i_embed

        x_u = F.relu(self.u_bn(self.u_layer1(nodes_u_embed)), inplace=True)
        x_u = F.dropout(x_u, training=self.training, p=self.droprate)
        x_u = self.u_layer2(x_u)

        x_i = F.relu(self.i_bn(self.i_layer1(nodes_i_embed)), inplace=True)
        x_i = F.dropout(x_i, training=self.training, p=self.droprate)
        x_i = self.u_layer2(x_i)

        x_ui = torch.cat((x_u, x_i), dim=1)
        x = F.relu(self.ui_bn(self.ui_layer1(x_ui)), inplace=True)
        x = F.dropout(x, training=self.training, p=self.droprate)

        scores = self.ui_layer2(x)
        return scores.squeeze()

    # 用于参数的正则化
    def regularization(self):
        regularization = 0
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        return regularization

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def loss(self, nodes_u, nodes_i, ratings):
        scores = self.forward(nodes_u, nodes_i)
        loss = self.criterion(scores, ratings)
        # total_loss = loss + self.regularization() + self.ssl_loss
        total_loss = loss + self.ssl_loss
        return total_loss


def train(model, train_loader, optimizer, epoch, rmse_mn, mae_mn, device):
    model.train()
    # model.train()非必须，但是若网络模型中含有dropout，batchnormal层的话则需要使用这一表达
    # model.eval()同上
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_u, batch_i, batch_ratings = data

        optimizer.zero_grad()
        loss = model.loss(batch_u.to(device), batch_i.to(device), batch_ratings.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()

        avg_loss += loss.item()

        # clamp the parameters
        layers = model.layers
        for k, layer in enumerate(layers):
            layer.constrain_parameters()

        if model.beta_ema > 0.:
            model.update_ema()
        if (i + 1) % 10 == 0:
            print('%s Training: [%d epoch, %3d batch] loss: %.5f, the best RMSE/MAE: %.5f / %.5f' % (
                datetime.now(), epoch, i + 1, avg_loss / 10, rmse_mn, mae_mn))
            avg_loss = 0.0
    return 0


def test(model, test_loader, device):
    model.eval()

    if model.beta_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    pred = []
    ground_truth = []

    for test_u, test_i, test_ratings in test_loader:
        test_u, test_i, test_ratings = test_u.to(device), test_i.to(device), test_ratings.to(device)
        scores = model(test_u, test_i)
        pred.append(list(scores.data.cpu().numpy()))
        ground_truth.append((list(test_ratings.data.cpu().numpy())))

    pred = np.array(sum(pred, []), dtype=np.float32)
    ground_truth = np.array(sum(ground_truth, []), dtype=np.float32)

    rmse = sqrt(mean_squared_error(pred, ground_truth))
    mae = mean_absolute_error(pred, ground_truth)

    if model.beta_ema > 0:
        model.load_params(old_params)
    return rmse, mae


def main():
    parser = argparse.ArgumentParser(description='sMCCF_yelp')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension')
    parser.add_argument('--weight_decay', type=int, default=0.0005, help='weight decay')
    parser.add_argument('--N', type=int, default=30000, help='L0 parameter')
    parser.add_argument('--droprate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=256, help='input batch size for testing')
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda or not')
    parser.add_argument('--load', type=bool, default=False, help='use checkpoint')
    parser.add_argument('--layer', type=int,default=3,help="be used for SSL")
    parser.add_argument('--aug_type', type=int, default=1, help="aug type")
    parser.add_argument('--ssl_reg', type=float, default=0.5, help="sslreg")
    parser.add_argument('--ssl_temp', type=float, default=0.5, help="ssltemp")
    parser.add_argument('--ssl_ratio', type=float, default=0.5, help="sslratio")
    args = parser.parse_args()

    print('Dataset: ' + args.dataset)
    print('-------------------- Hyperparams --------------------')
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))

    device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    embed_dim = args.embed_dim
    data_path = './datasets/' + args.dataset

    print("改动：\n"
          "多图嵌入为att\n"
          "memery数改为4\n"
          "combiner改为attention")

    with open(data_path + '/_allData.p', 'rb') as meta:
        u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj = pickle.load(meta)
        # I2E:2614*1286  U2E:1286*2614
    with open(data_path + '/friends.p', 'rb') as meta2:
        u_friends, i_friends = pickle.load(meta2)

        '''===========================change=============================='''
    dataset = Loader(args.dataset, device)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(dataset.trainUser), torch.LongTensor(dataset.trainItem),
                                                  torch.FloatTensor(dataset.trainRating))

    testset = torch.utils.data.TensorDataset(torch.LongTensor(dataset.testUser), torch.LongTensor(dataset.testItem),
                                                 torch.FloatTensor(dataset.testRating))

    '''===========================change=============================='''

    # trainset = torch.utils.data.TensorDataset(torch.LongTensor(u_train), torch.LongTensor(i_train),
    #                                           torch.FloatTensor(r_train))
    #
    # testset = torch.utils.data.TensorDataset(torch.LongTensor(u_test), torch.LongTensor(i_test),
    #                                          torch.FloatTensor(r_test))

    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=16, pin_memory=True)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=16, pin_memory=True)
    # print(type(u2e.to(device).num_embeddings)) # <class 'int'>
    '''===========================对U-I加入SSL处理=================================='''
    '''加入SSL学习得到的联合任务的损失，需要改变原user part的u2e，i2e，用ssl计算获得'''
    ssl_loss=SSL(u2e.to(device), i2e.to(device), dataset, embed_dim, args.layer, args.aug_type ,args.ssl_reg, args.ssl_temp, args.ssl_ratio)

    # user part
    u_agg_embed_cmp1 = aggregator(u2e.to(device), i2e.to(device), u_adj, embed_dim, device=device,
                                  weight_decay=args.weight_decay, droprate=args.droprate)
    u_embed_cmp1 = encoder(embed_dim, u_agg_embed_cmp1, device=device)
    u_memory = memory(u2e.to(device), i2e.to(device), u_friends, embed_dim, weight_decay=args.weight_decay,
                      droprate=args.droprate, device=device)
    '''融合函数'''
    u_embed = combiner(u_embed_cmp1, u_memory, embedding_dim=embed_dim, droprate=args.droprate, device=device,
                       is_user_part=True)

    # item part
    i_agg_embed_cmp1 = aggregator(u2e.to(device), i2e.to(device), i_adj, embed_dim, device=device,
                                  weight_decay=args.weight_decay, droprate=args.droprate, is_user_part=False)
    i_embed_cmp1 = encoder(embed_dim, i_agg_embed_cmp1, device=device, is_user_part=False)
    i_memory = memory(u2e.to(device), i2e.to(device), i_friends, embed_dim, weight_decay=args.weight_decay,
                      droprate=args.droprate, device=device, is_user_part=False)
    i_embed = combiner(i_embed_cmp1, i_memory, embedding_dim=embed_dim, droprate=args.droprate, device=device,
                       is_user_part=False)



    # model
    model = SMCCF(u_embed, i_embed, ssl_loss, dataset, embed_dim, args.N, droprate=args.droprate, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rmse_mn = np.inf
    mae_mn = np.inf
    endure_count = 0
    start_epoch = 0
    if args.load:
        path_checkpoint = "./checkpoints/ckpt_best_1.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # ======================= training  ===============
        train(model, _train, optimizer, epoch, rmse_mn, mae_mn, device)
        # ======================= test  ===============
        rmse, mae = test(model, _test, device)

        if rmse_mn > rmse:
            rmse_mn = rmse
            mae_mn = mae
            endure_count = 0
        else:
            endure_count += 1

        print("<Test> RMSE: %.5f, MAE: %.5f " % (rmse, mae))

        if endure_count > 60:
            break

        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        if not os.path.isdir("./checkpoints"):
            os.mkdir("./checkpoints")
        torch.save(checkpoint, './checkpoints/ckpt_best_%s.pth' % (str(epoch)))

    print('The best RMSE/MAE: %.5f / %.5f' % (rmse_mn, mae_mn))


if __name__ == '__main__':
    main()
