import argparse  # 导入命令行参数解析库
import torch  # 导入PyTorch库
from torch.utils.data import DataLoader
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用的功能性神经网络操作
import torch.optim as optim  # 导入优化器模块
from torchvision import datasets, transforms  # 导入数据集和转换工具
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器

# 定义一个卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层和全连接层
        self.linear = nn.Linear(784,128)
        
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 第一卷积层，输入1个通道，输出32个通道，卷积核大小为3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 第二卷积层
        self.dropout1 = nn.Dropout(0.25)  # 第一个Dropout层，防止过拟合
        self.dropout2 = nn.Dropout(0.5)  # 第二个Dropout层
        self.fc1 = nn.Linear(9216, 128)  # 第一个全连接层
        
        self.fc2 = nn.Linear(256, 10)  # 第二个全连接层，输出10个类别

    def forward(self, x): # x [64, 1, 28, 28]
        y = torch.flatten(x, 1) # torch.Size([64, 784])
        y = self.linear(y) # torch.Size([64, 128])
        y = F.relu(y)  # torch.Size([64, 128])
        
        x = self.conv1(x)  # torch.Size([64, 32, 26, 26])
        x = F.relu(x)  # torch.Size([64, 32, 26, 26])
        x = self.conv2(x)  # torch.Size([64, 64, 24, 24])
        x = F.relu(x)  # torch.Size([64, 64, 24, 24])
        x = F.max_pool2d(x, 2)  # torch.Size([64, 64, 12, 12])
        x = self.dropout1(x)  # torch.Size([64, 64, 12, 12])
        x = torch.flatten(x, 1)  # torch.Size([64, 9216])
        x = self.fc1(x)  # torch.Size([64, 128])
        x = F.relu(x)  # torch.Size([64, 128])
        x = self.dropout2(x)  # torch.Size([64, 128])
        
        # 将 output1 和 output2 连接在一起
        combined = torch.cat((x, y), dim=1)  # torch.Size([64, 256])

        x = self.fc2(combined)  # torch.Size([64, 10])
        output = F.log_softmax(x, dim=1)  # torch.Size([64, 10])
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # 设置模型为训练模式，启用 Dropout 和 Batch Normalization
    for batch_idx, (data, target) in enumerate(train_loader):  # 遍历训练数据加载器
        data, target = data.to(device), target.to(device)  # 将数据和目标转移到指定设备（GPU 或 CPU）
        optimizer.zero_grad()  # 清除之前的梯度，以防止累加
        output = model(data)  # 前向传播，获取模型输出
        loss = F.nll_loss(output, target)  # 计算负对数似然损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        
        # 每隔一定批次打印训练状态
        if batch_idx % args.log_interval == 0:  # 检查是否达到记录状态的间隔
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),  # 输出当前 epoch 和已处理样本数量
                100. * batch_idx / len(train_loader),  # 计算并输出当前进度百分比
                loss.item()))  # 输出当前批次的损失值
            if args.dry_run:  # 如果是快速运行模式
                break  # 提前退出训练过程


def test(model, device, test_loader):
    model.eval()  # 设置模型为评估模式，此时会关闭 Dropout 和 Batch Normalization
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测的数量
    with torch.no_grad():  # 在此上下文中，不计算梯度，节省内存和计算资源
        for data, target in test_loader:  # 遍历测试数据加载器
            data, target = data.to(device), target.to(device)  # 将数据和目标转移到指定设备（GPU 或 CPU）
            output = model(data)  # 前向传播，获取模型输出
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 计算负对数似然损失并累加
            pred = output.argmax(dim=1, keepdim=True)  # 获取每个样本输出概率最大的类别索引作为预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计与目标匹配的正确预测数量

    test_loss /= len(test_loader.dataset)  # 计算平均损失（总损失除以测试集样本数量）

    # 打印测试结果，包括平均损失和准确率
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),  # 输出平均损失、正确预测数量和测试集样本数量
        100. * correct / len(test_loader.dataset)))  # 输出准确率（正确预测数量除以总样本数量）


def main():
    # 训练设置
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')  # 创建参数解析器
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')  # 训练时的批次大小
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')  # 测试时的批次大小
    parser.add_argument('--epochs', type=int, default=1400, metavar='N',
                        help='number of epochs to train (default: 1400)')  # 训练的迭代次数
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')  # 学习率
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')  # 学习率调度的参数
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  # 禁用CUDA训练的标志
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')  # 禁用macOS GPU训练的标志
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')  # 快速检查单次训练迭代的标志
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')  # 随机种子，确保实验可重复性
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')  # 记录训练状态的间隔批次
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')  # 保存当前模型的标志
    args = parser.parse_args()  # 解析命令行参数

    # 确定使用的设备（CUDA或MPS或CPU）
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)  # 设置随机种子

    # 选择设备
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 设置数据加载器的参数
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 数据归一化
    ])
    
    # 加载MNIST数据集
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                               transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                               transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)  # 创建模型并转移到设备
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # 定义优化器

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # 定义学习率调度器
    
    # 训练和测试模型
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)  # 训练
        test(model, device, test_loader)  # 测试
        scheduler.step()  # 更新学习率

        # 保存模型
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()  # 运行主函数