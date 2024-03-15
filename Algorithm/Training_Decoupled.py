#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
from tqdm import tqdm

from models import ResNet18_cifar
from models import ResNet18_widar
from models.Fed import Aggregation, get_model_list
from models.vgg import vgg_16_bn
from utils.Clients import Clients
from utils.utils import save_result, my_save_result, get_final_acc
from models.test import test_img, test
from models.Update import DatasetSplit, LocalUpdate_FedAvg
from optimizer.Adabelief import AdaBelief


def Decoupled(args, dataset_train, dataset_test, dict_users):
    net_glob_list, net_slim_info = get_model_list(args)

    # training
    acc_list = [[] for _ in net_glob_list]

    my_list = list(map(float, args.client_hetero_ration.split(':')))
    hetero_proportion = [round(x / sum(my_list), 2) for x in my_list]

    # 开始训练
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = [[] for _ in net_glob_list]
        lens = [[] for _ in net_glob_list]
        idxs_users = [[] for _ in net_glob_list]

        m = max(int(args.frac * args.num_users), 1)
        for id in range(len(net_glob_list)):
            idxs_users[id] = np.random.choice(range(int(args.num_users * sum(hetero_proportion[:id])), args.num_users),
                                              m, replace=False)
        for id in range(len(idxs_users)):
            print(f"user {id}: this epoch choose: {idxs_users[id]}")
        print(f"hetero_proportion: {hetero_proportion}")
        # 需要print 每个客户端的计算资源

        for id, users in enumerate(idxs_users):
            for idx in users:
                local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w = local.train(round=iter, net=copy.deepcopy(net_glob_list[id]).to(args.device))  # 这里开始正式训练
                w_locals[id].append(copy.deepcopy(w))
                lens[id].append(len(dict_users[idx]))

            # update global weights
            w_glob = Aggregation(w_locals[id], lens[id])
            net_glob_list[id].load_state_dict(w_glob)
            print(net_slim_info[id])
            acc_list[id].append(test(net_glob_list[id], dataset_test, args))

    for id, acc in enumerate(acc_list):
        file = my_save_result(acc, str(net_slim_info[id]), 'acc', args)
    get_final_acc(file)
