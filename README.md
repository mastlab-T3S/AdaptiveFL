# AdaptiveFL: Adaptive Heterogeneous Federated Learning for Resource-Constrained AIoT Systems

Code for the following paper:

Chentao Jia, Ming Hu, Zekai Chen, Yanxin Yang, Xiaofei Xie, Yang Liu, and Mingsong Chen, "[AdaptiveFL: Adaptive Heterogeneous Federated Learning for Resource-Constrained AIoT Systems](https://arxiv.org/abs/2311.13166)," ACM Design Automation Conference (DAC), San Francisco, CA, USA, Jun. 24-27, 2024.

Note: The scripts will be slow without the implementation of parallel computing. 


## Introduction
Although Federated Learning (FL) is promising to enable collaborative learning among Artificial Intelligence of Things (AIoT) devices, it suffers from the problem of low classification performance due to various heterogeneity factors (e.g., computing capacity, memory size) of devices and uncertain operating environments. To address these issues, this paper introduces an effective FL approach named AdaptiveFL based on a novel fine-grained width-wise model pruning mechanism, which can generate various heterogeneous local models for heterogeneous AIoT devices. By using our proposed reinforcement learning-based device selection strategy, AdaptiveFL can adaptively dispatch suitable heterogeneous models to corresponding AIoT devices based on their available resources for local training. Experimental results show that, compared to state-of-the-art methods, AdaptiveFL can achieve up to 16.83% inference improvements for both IID and non-IID scenarios. 

<img src="https://s2.loli.net/2024/03/15/5v38u4fIeLXdbBS.png" alt="3" style="zoom:25%;" />

## Requirements
python>=3.7  

pytorch>=2.1

## Run

AdaptiveFL:
> python [main_fed.py](main_fed.py) --gpu 0 --algorithm AdaptiveFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --depth_saved 4 6 8 --width_ration 0.4 0.66 1.0 --client_chosen_mode available

> python [main_fed.py](main_fed.py) --gpu 0 --algorithm AdaptiveFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --depth_saved 4 6 8 --width_ration 0.4 0.66 1.0 --client_chosen_mode available

HeteroFL:
> python [main_fed.py](main_fed.py) --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --depth_saved 0 --width_ration 0.5 0.71 1.0 --client_chosen_mode available

> python [main_fed.py](main_fed.py) --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --depth_saved 0 --width_ration 0.5 0.71 1.0 --client_chosen_mode available

Decoupled:
> python [main_fed.py](main_fed.py) --gpu 0 --algorithm Decoupled --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --depth_saved 8 --width_ration 0.4 0.66 1.0 --client_chosen_mode available

> python [main_fed.py](main_fed.py) --gpu 0 --algorithm Decoupled --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --depth_saved 8 --width_ration 0.4 0.66 1.0 --client_chosen_mode available


See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

NB: for CIFAR-10, `num_channels` must be 3.

NB: for Widar, `num_channels` and `num_classes` must be 22.


--algorithm AdaptiveFL --model mobilenet --dataset widar --num_users 17 --depth_saved 2 3 4 --width_ration 0.46 0.69 1.0 --iid 0 --noniid_case 5 --data_beta 0.2 --client_hetero_ration 4:10:3 --frac 0.6

## Results

![image-20240315102602639](https://s2.loli.net/2024/03/15/arwh5FyneVOWotb.png)

<center>
    <img src="https://s2.loli.net/2024/03/15/s72ldS4XCfgT9pw.png">
</center>

<center>
        <img src="https://s2.loli.net/2024/03/15/Gmhg5Y62cKSP4kv.png">
</center>



## Acknowledgements
Acknowledgements give to [youkaichao](https://github.com/youkaichao).

## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.


Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine Learning and Systems 2 (2020): 429-450.


Li, Qinbin, Bingsheng He, and Dawn Song. "Model-contrastive federated learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

Fraboni, Yann, et al. "Clustered sampling: Low-variance and improved representativity for clients selection in federated learning." International Conference on Machine Learning. PMLR, 2021.

Yao, Dezhong, et al. "Local-Global Knowledge Distillation in Heterogeneous Federated Learning with Non-IID Data." arXiv preprint arXiv:2107.00051 (2021).

Zhu, Zhuangdi, Junyuan Hong, and Jiayu Zhou. "Data-free knowledge distillation for heterogeneous federated learning." International Conference on Machine Learning. PMLR, 2021.

Gao, Liang, et al. "FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

Kim, Jinkyu and Kim, Geeho and Han, Bohyung. "Multi-Level Branched Regularization for Federated Learning." International Conference on Machine Learning. PMLR, 2022.

Lee, Gihun, et al. "Preservation of the global knowledge by not-true distillation in federated learning." Advances in Neural Information Processing Systems 35 (2022): 38461-38474.