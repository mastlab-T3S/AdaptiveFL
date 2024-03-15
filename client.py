from typing import Tuple

import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader

from utils.ConnectHandler_client import ConnectHandler
from utils.hetero_client_config import HeteroClient, LOW_RESOURCE
from utils.options import args_parser
from utils.set_seed import set_random_seed
from utils.widar import WidarDataset

HIGH_RESOURCE = {'loc': 10, 'scale': 0.7, 'up_limit': 8, 'low_limit': 12}
MEDIUM_RESOURCE = {'loc': 4.5, 'scale': 1, 'up_limit': 25, 'low_limit': 19}
LOW_RESOURCE = {'loc': 11, 'scale': 1, 'up_limit': 14, 'low_limit': 9}

class WidarDataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.targets = [d[1] for d in self.data]

        self.classes = list(range(22))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.data[idx][0].reshape(22, 20, 20), self.data[idx][1]


'''
返回的参数：
    模型权重
    本地数据量
    返回模型的下标
'''
if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)
    ID = 0

    RESOURCE_PERFORMANCE = LOW_RESOURCE

    client = HeteroClient(RESOURCE_PERFORMANCE['loc'], RESOURCE_PERFORMANCE['scale'], RESOURCE_PERFORMANCE['up_limit'], RESOURCE_PERFORMANCE['low_limit'])

    data =[]
    with open(r'./data/widar/1.pkl', 'rb') as f:
        print(f)
        data.append(torch.load(f))
    x = [d[0] for d in data]
    x = np.concatenate(x, axis=0, dtype=np.float32)
    x = (x - .0025) / .0119
    y = np.concatenate([d[1] for d in data])
    data = [(x[i], y[i]) for i in range(len(x))]

    dataset_train = WidarDataset(data)
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)
    loss_func = nn.CrossEntropyLoss()
    ldr_train = DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True, drop_last=True)

    while True:
        recv = connectHandler.receiveFromServer()  # net  net_idx

        net = recv['net']
        requirement = recv['net_size']
        algorithm = recv['algorithm']

        # if algorithm == 'FedSlim':
        #     real_resource = client.get_resource()
        #     if real_resource >= requirement:  # accept
        #         feedback_model_idx = model_idx
        #     else:  # reject
        #         feedback_model_idx = -1
        #         for i in range(len(model_size) - 1):
        #             if model_size[i + 1] > real_resource:
        #                 feedback_model_idx = i
        #                 break

        net.train()

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr * (args.lr_decay ** round),
                                        momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        Predict_loss = 0
        for iter in range(args.local_ep):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)
                if args.dataset == 'widar':
                    labels = labels.long()
                net.zero_grad()
                log_probs = net(images)['output']
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                Predict_loss += loss.item()

        info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (args.local_ep * len(ldr_train)))
        print(info)

        return_dict = {}
        return_dict['local_w'] = net.state_dict()
        # return_dict['local_net_idx'] = feedback_model_idx
        return_dict['local_len'] = len(dataset_train)
        connectHandler.uploadToServer(return_dict)
