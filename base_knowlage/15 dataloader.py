import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("./Dataset/CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# drop_last=False时，当前batch不足size时仍然保留
# shuffle=True时，每次选取的顺序都是打乱的

# 测试数据集中第一张图片及label
img, target = test_data[0]
writer = SummaryWriter("test_loader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # 这里的image就是一个batch
        # print(imgs.shape)
        # print(target)
        writer.add_images("test_image{}".format(epoch), imgs, step)
        step += 1

writer.close()


