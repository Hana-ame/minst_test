import argparse  # 导入命令行参数解析库
import torch  # 导入PyTorch库
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
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 第一卷积层，输入1个通道，输出32个通道，卷积核大小为3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 第二卷积层
        self.dropout1 = nn.Dropout(0.25)  # 第一个Dropout层，防止过拟合
        self.dropout2 = nn.Dropout(0.5)  # 第二个Dropout层
        self.fc1 = nn.Linear(9216, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 10)  # 第二个全连接层，输出10个类别

    def forward(self, x):
        # 定义前向传播过程
        x = self.conv1(x)  # 通过第一卷积层
        x = F.relu(x)  # 使用ReLU激活函数
        x = self.conv2(x)  # 通过第二卷积层
        x = F.relu(x)  # 使用ReLU激活函数
        x = F.max_pool2d(x, 2)  # 最大池化层，降低维度
        x = self.dropout1(x)  # 第一个Dropout层
        x = torch.flatten(x, 1)  # 展平多维输入为一维
        x = self.fc1(x)  # 通过第一个全连接层
        x = F.relu(x)  # 使用ReLU激活函数
        x = self.dropout2(x)  # 第二个Dropout层
        x = self.fc2(x)  # 通过第二个全连接层
        output = F.log_softmax(x, dim=1)  # 使用log_softmax激活函数，输出为类别的对数概率
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 将数据和目标转移到指定设备
        optimizer.zero_grad()  # 清除之前的梯度
        output = model(data)  # 前向传播
        loss = F.nll_loss(output, target)  # 计算负对数似然损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        # 每隔一定批次打印训练状态
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break  # 如果是快速运行，提前退出


def test(model, device, test_loader):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 转移数据到设备
            output = model(data)  # 前向传播
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的索引作为预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确预测的数量

    test_loss /= len(test_loader.dataset)  # 计算平均损失

    # 打印测试结果
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # 训练设置
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1400, metavar='N',
                        help='number of epochs to train (default: 1400)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
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
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

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