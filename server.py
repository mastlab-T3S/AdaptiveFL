from utils.ConnectHandler_server import ConnectHandler
from utils.options import args_parser
from utils.set_seed import set_random_seed
import time

import torch
import numpy as np
from tqdm import tqdm

from models.Fed import get_model_list, Aggregation_FedSlim, split_model, select_clients

from utils.utils import save_result, my_save_result, get_final_acc
from models.test import test_img, test

'''
模型
算法名
模型大小
'''


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)

    net_glob_list, net_slim_info = get_model_list(args)

    # training
    total_time = 0
    time_list = []
    acc_list = [[] for _ in net_glob_list]

    # 开始训练
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        returnModel_idx = []

        m = max(int(args.frac * args.num_users), 1)
        connectHandler = ConnectHandler(args.num_users, args.HOST, args.POST)  # 会阻塞 ，server 的ip 和端口

        ration_users = np.random.choice(range(len(net_glob_list)), m)  # 模型选择
        idx_users = select_clients(args, ration_users, len(net_glob_list))  # 基于规则的客户端选择

        feedback_list = []
        begin = time.time()
        for id, idx in enumerate(idx_users):
            sent_data = {}
            sent_data['net'] = net_glob_list[ration_users[id]]
            connectHandler.sendData(idx, sent_data)  # 发数据

        while len(w_locals) < len(idx_users):
            recv_data = connectHandler.receiveData()  # 收数据
            w_locals.append(recv_data['local_w'])
            lens.append(recv_data['local_len'])
            returnModel_idx.append(recv_data['local_net_idx'])

        train_time = time.time() - begin
        total_time += train_time
        time_list.append(total_time)
        print(f"this epoch cost time:{train_time}")
        print(f"this epoch choose: {idx_users}")  # 这一轮选择的用户下标
        print(f"this epoch dispatch models: {ration_users}")  # 每个用户对应的模型的比例, 初始分发
        print(f"this epoch received models: {feedback_list}")  # 每个用户对应的模型的比例, 初始分发

        w_glob_param = Aggregation_FedSlim(w_locals, lens, net_glob_list[-1].state_dict())

        for idx, net in enumerate(net_glob_list):
            net.load_state_dict(split_model(w_glob_param, net.state_dict()))
            print(net_slim_info[idx])
            acc_list[idx].append(test(net, dataset_test, args))

    save_result(time_list, 'test_time', args)
    for id, acc in enumerate(acc_list):
        file = my_save_result(acc, str(net_slim_info[id]), 'acc', args)
        # save_result(acc, 'accuracy', args)
    get_final_acc(file)
