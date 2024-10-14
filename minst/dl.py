import argparse  # 导入命令行参数解析库
import torch  # 导入PyTorch库
from torch.utils.data import DataLoader
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用的功能性神经网络操作
import torch.optim as optim  # 导入优化器模块
from torchvision import datasets, transforms  # 导入数据集和转换工具
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器

from typing import Tuple
from main import Net as MinstModel

# 设置数据加载器的参数
train_kwargs = {'batch_size': 64}
test_kwargs = {'batch_size': 1000}

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 数据归一化
])


# 创建模型实例
model: nn.Module = MinstModel()

# 从文件加载模型
model.load_state_dict(torch.load("mnist_cnn.pt"))  # 加载模型参数
model.eval()  # 设置模型为评估模式


# 加载MNIST数据集
dataset1 = datasets.MNIST('../data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                          transform=transform)
train_loader = DataLoader(dataset1, **train_kwargs)
test_loader: DataLoader[Tuple[torch.Tensor, int]] = DataLoader(dataset2, **test_kwargs)


test_loss = 0  # 初始化测试损失
correct = 0  # 初始化正确预测的数量
for data, target in test_loader:  # 遍历测试数据加载器
    data: torch.Tensor
    target: torch.Tensor
    output = model(data)  # 前向传播，获取模型输出
    test_loss += F.nll_loss(output, target, reduction='sum').item()  # 计算负对数似然损失并累加
    pred = output.argmax(dim=1, keepdim=True)  # 获取每个样本输出概率最大的类别索引作为预测结果
    correct += pred.eq(target.view_as(pred)).sum().item()  # 统计与目标匹配的正确预测数量
    break
