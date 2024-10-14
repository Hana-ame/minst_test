import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # 返回数据和对应的标签
        return self.data[index], index  # 这里假设标签就是索引

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

# 创建数据集实例
data = [1, 2, 3, 4, 5]
dataset = MyDataset(data)

# 创建数据加载器
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 遍历数据加载器
for batch_data, batch_labels in dataloader:
    print("Batch Data:", batch_data)
    print("Batch Labels:", batch_labels)
    print("---")