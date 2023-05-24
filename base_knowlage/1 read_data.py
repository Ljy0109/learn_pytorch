from torch.utils.data import Dataset
from PIL import Image ##
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):  # 为整个类提供一些全局变量
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)  # 将根路径和类别路径 拼接
        self.img_path = os.listdir(self.path)  # 使用列表存储path下的所有文件名

    def __getitem__(self, index):
        """
        返回指定index的图片和label
        """
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)  # 使用Image读取图片
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "DataSet/hymenoptera_data/hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)  # 创建类的示例 img, label = ants_dataset[index]
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset  # 可以直接相加用于拼接 dataset的集合
