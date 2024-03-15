#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
from tqdm import tqdm

from Algorithm.Training_AdaptiveFL import Aggregation_AdaptiveFL, split_model
from models import ResNet18_cifar, MobileNetV2
from models import ResNet18_widar
from models.Fed import get_model_list, select_clients
from models.vgg import vgg_16_bn
from utils.Clients import Clients
from utils.utils import save_result, my_save_result, get_final_acc
from models.test import test_img, test
from models.Update import DatasetSplit, LocalUpdate_FedAvg
from optimizer.Adabelief import AdaBelief



def HeteroFL(args, dataset_train, dataset_test, dict_users):
    net_glob_list, net_slim_info = get_model_list(args)

    # training
    acc_list = [[] for _ in net_glob_list]

    # 开始训练
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []

        m = max(int(args.frac * args.num_users), 1)
        ration_users = np.random.choice(range(len(net_glob_list)), m, replace=True)  # 模型选择
        idx_users = select_clients(args, ration_users, len(net_glob_list))

        print(f"this epoch choose: {idx_users}")
        print(f"this epoch models: {ration_users}")
        print(f"hetero_proportion: \t{args.client_hetero_ration}")
        # 需要print 每个客户端的计算资源

        for id, idx in enumerate(idx_users):
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter,
                            net=copy.deepcopy(net_glob_list[ration_users[id]]).to(args.device))  # 这里开始正式训练

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

        w_glob = Aggregation_AdaptiveFL(w_locals, lens, net_glob_list[-1].state_dict())

        for idx, net in enumerate(net_glob_list):
            net.load_state_dict(split_model(w_glob, net.state_dict()))
            print(net_slim_info[idx])
            acc_list[idx].append(test(net, dataset_test, args))

    for id, acc in enumerate(acc_list):
        file = my_save_result(acc, str(net_slim_info[id]), 'acc', args)
    get_final_acc(file)


