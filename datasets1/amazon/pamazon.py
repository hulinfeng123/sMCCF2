# -*- coding: utf-8 -*-
"""
@author: liulin
@time: 2021/7/6 15:13
@subscribe: 

"""

# -*- coding: utf-8 -*-
import pickle
import random


def loaduAdj():
    u_adj = {}
    with open('./user_item.txt') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [i for i in l[1:]]
                uid = int(l[0])
                u_adj[uid] = items
    return u_adj


def createUU():
    u_adj = loaduAdj()

    u_neigh = {}
    for key in u_adj.keys():
        u_nei = {}
        for key2 in u_adj.keys():
            if key2 != key:
                count = 0
                for item in u_adj[key]:
                    if item in u_adj[key2]:
                        count += 1
                if count > 0:
                    u_nei[key2] = count
        u_neigh[key] = sorted(u_nei.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:60]
        # print(str(key) + " ", u_neigh[key])

        nei = [item[0] for item in u_neigh[key]]

        print(key, end=" ")
        for i in nei:
            print(i, end=" ")

        print()


def createII():
    u_adj = loaduAdj()
    i_adj = {}
    for i in range(1500):
        i_a = []
        for key in u_adj.keys():
            if str(i) in u_adj[key]:
                i_a.append(int(key))

        i_adj[i] = i_a
    #

    i_neigh = {}
    for key in i_adj.keys():
        i_nei = {}
        for key2 in i_adj.keys():
            if key2 != key:
                count = 0
                for user in i_adj[key]:
                    if user in i_adj[key2]:
                        count += 1
                if count > 0:
                    i_nei[key2] = count
        i_neigh[key] = sorted(i_nei.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:60]
        # print(str(key) + " ", i_neigh[key])

        nei = [item[0] for item in i_neigh[key]]

        print(key, end=" ")
        for i in nei:
            print(i, end=" ")

        print()


def creatadj():
    u_adj = {}
    with open('./uu.txt') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                neigh = [int(i) for i in l[1:]]
                uid = int(l[0])
                u_adj[uid] = neigh

    i_adj = {}
    with open('./ii.txt') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                neigh = [int(i) for i in l[1:]]
                iid = int(l[0])
                i_adj[iid] = neigh

    with open('./neighs.p', 'wb') as meta:
        pickle.dump((u_adj, u_adj), meta)


def printadj(adj):
    for key in adj.keys():
        print(key, adj[key])


def get_data():
    with open('./neighs2.p', 'rb') as meta:
        u_neighs, i_neighs = pickle.load(meta)

    for index in u_neighs.keys():
        print(index, ":", u_neighs[index])


def spiltDataSet():
    data = []
    for line in open('./user_item.dat', 'r'):
        (user, item, rating, time) = line.split('	')
        if int(user) < 1000 and int(item) < 1000:
            data.append((user, item, rating))

    random.shuffle(data)

    length = data.__len__()

    data_train = data[:int(length * 0.8)]
    data_test = data[int(length * 0.8):]

    for da in data_train:
        print(da[0] + " " + da[1] + " " + da[2])

    print('------------------test--------------------')

    for da in data_test:
        print(da[0] + " " + da[1] + " " + da[2])


def allData():
    u_train = []
    i_train = []
    r_train = []

    u_test = []
    i_test = []
    r_test = []

    for line in open('train.txt', 'r'):
        (user, item, rating) = line.split(' ')
        u_train.append(user)
        i_train.append(item)
        r_train.append(rating)

    for line in open('test.txt', 'r'):
        (user, item, rating) = line.split(' ')
        u_test.append(user)
        i_test.append(item)
        r_test.append(rating)

    with open('./_allData.p', 'wb') as meta:
        pickle.dump((u_train, i_train, r_train, u_test, i_test, r_test), meta)


def transform(filename):
    inter = {}
    for line in open(filename, 'r'):
        (user, item, rating) = line.split(' ')
        if int(user) < 1000 and int(item) < 1000:
            if int(user) in inter.keys():
                inter[int(user)].append(int(item))
            else:
                inter[int(user)] = [int(item)]

    maxitem = -1
    intercount = 0

    for i in sorted(inter):
        print(i, end=" ")
        for inte in inter[i]:
            intercount += 1
            if inte > maxitem:
                maxitem = inte
            print(inte, end=" ")
        print()

    print("maxitem: ", maxitem)
    print("intercount: ", intercount)


if __name__ == '__main__':
    spiltDataSet()
