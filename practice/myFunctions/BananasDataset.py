"""
一个用于加载香蕉检测数据集的自定义数据集
"""
import torch
from torch.utils.data import Dataset  # Dataset是抽象类，需要重定义
from torch.utils.data import DataLoader
import os
import pandas as pd
import torchvision


def read_data_bananas(data_dir, is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    # 将'img_name'这一列设置为索引列
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    # for index, row in csv_data.iterrows(): 迭代遍历csv_data
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(  # 该方法读取图片会直接转为(C, H, W)的张量
            os.path.join(data_dir, 'bananas_train' if is_train else
            'bananas_val', 'images', f'{img_name}')))  # f'{img_name}'是将花括号的内容作为字符串插入
        # 这⾥的target包含（类别，左上⻆x，左上⻆y，右下⻆x，右下⻆y），
        # 其中所有图像都具有相同的⾹蕉类（索引为0）
        targets.append(list(target))  # target是pandas的Series类型,可以用list换成列表
    return images, torch.tensor(targets).unsqueeze(1) / 256  # 将真实框的坐标归一化

# data_dir = 'E:\CODE\Python\PyCharm\Project\learn_pytorch\practice\dataset\\banana-detection\\banana-detection'
# imgs, targets = read_data_bananas(data_dir=data_dir ,is_train=True)
# imgs是一个list，大小为1000表示有1000张图
# targets是(1000, m, 5)的张量，m表示一张图有几个目标，这个数据集只有一个，所以m=1


class BananasDataset(Dataset):
    """
    魔法方法与在类中定义的普通函数之间有几个关键区别：
    命名方式：魔法方法是以双下划线开头和结尾的特殊命名方式，例如__init__、__str__。而普通函数可以使用任意合法的函数名。
    调用方式：魔法方法会在特定的情况下自动调用，而普通函数需要手动调用。例如，__init__方法在创建对象时自动调用，而普通函数需要在需要的时候手动调用。
    定义作用：魔法方法用于定义类的特定行为和操作，例如初始化、比较、运算符重载等。而普通函数用于实现类的其他功能，可能与特定行为和操作无关。
    内部机制：魔法方法与 Python 解释器密切相关，定义了类的内部行为。普通函数则是类的一部分，但与解释器的内部机制无关。
    总的来说，魔法方法是一种特殊的方法，用于定义类的特定行为和操作，而普通函数则用于实现其他功能。魔法方法通过特定的命名方式和自动调用机制，与解释器紧密配合，使得自定义类具有类似内置类型的行为。
    """
    def __init__(self, filepath, is_train):
        self.imgs, self.targets = read_data_bananas(filepath, is_train)
        print('read' + str(len(self.imgs)) + (f'training examples' if is_train else f'validation examples'))

    def __getitem__(self, index):
        # 提取指定索引的数据
        return (self.imgs[index].float(), self.targets[index])

    def __len__(self):
        # 返回数据的长度
        return len(self.imgs)


def load_data_bananas(data_dir, batch_size):
    """加载香蕉数据集，分别加载训练集和测试集"""
    train_dataset = DataLoader(BananasDataset(data_dir, is_train=True),
                               batch_size=batch_size,
                               shuffle=True)
    val_dataset = DataLoader(BananasDataset(data_dir, is_train=False),
                             batch_size=batch_size,
                             shuffle=False)
    return train_dataset, val_dataset


if __name__ == '__main__':
    data_dir = 'E:\CODE\Python\PyCharm\Project\learn_pytorch\practice\dataset\\banana-detection\\banana-detection'
    # dataset = BananasDataset(data_dir, )
    train_dataset, val_dataset = load_data_bananas(data_dir, batch_size=64)

    batch = next(iter(train_dataset))
