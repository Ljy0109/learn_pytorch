"""
实现SSDnet
本次实现的SSDnet由五个模块组成。
每个模块分为三部分：
    一部分使用正常的res卷积网络提取特征
    一部分根据特征图预测锚框的类别 -> cls_predictor()
    一部分根据特征图预测锚框的偏移量 -> bbox_predictor()
其中提取特征的res卷积网络模块为了实现不同尺度的锚框预测，额外设定了降采样池化

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

from myFunctions.BananasDataset import load_data_bananas
from myFunctions.anchor_box import multibox_prior
import myFunctions.anchor_get_target
from myFunctions.anchor_get_target import multibox_target


def cls_predictor(num_inputs, num_anchors, num_classes):
    # 使用卷积层预测锚框对应的类型
    # num_inputs是特征图的通道数
    # 预测锚框类型需要给出它属于所有类型（包括背景）的概率（分数）
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    # 使用卷积层预测锚框的偏移量
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    # 自定义一个前向传播函数，用于测试单个模块block的输出
    return block(x)

# Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
# Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
# Y1.shape, Y2.shape
# cls_predictor 和 bbox_predictor 不会改变输出特征图的尺寸，但是神经网络中的特征图尺寸是在不断变化的
# SSDnet中需要把所有的（不同尺度的）cls_predictor 和 bbox_predictor整合到一起 变成cls_pre 和 bbox_pre用于损失值的计算
# 所以需要对输出的预测值的后三个通道进行展平处理


def flatten_pred(pred):
    # 对预测值进行展平处理.permute(0, 2, 3, 1)将原来的(0, 1, 2, 3)维度排列改为(0, 2, 3, 1)
    # 从dim=1维度开始展平后面的维度
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    # 按照batch_size维度整合到一起
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# concat_preds([Y1, Y2]).shape


def down_sample_blk(in_channels, out_channels):
    # 实现正常的二层ResNet模块，然后加上最大池化缩小一半的尺寸
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    # 该网络串联3个高和宽减半块，并逐步将通道数翻倍。
    # 给定输入图像的形状为256x256，此基本网络块输出的特征图形状为32x32(256 / (2^3))。
    blk = []
    num_filters = [3, 16, 32, 64]  # 输出通道数逐步翻倍，最后输出64通道
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# forward(torch.zeros((2, 3, 256, 256)), base_net()).shape


def get_blk(i):
    # 总的SSDnet模块组成顺序
    if i == 0:
        blk = base_net()
    elif i == 1:  # base_net的输出通道是64
        blk = down_sample_blk(64, 128)
    elif i == 4:
        # nn.AdaptiveMaxPool2d((1, 1)) 表示一个自适应最大池化层，其中 (1, 1) 是输出的目标大小，表示希望输出的特征图大小为 (1, 1)。
        # 这意味着池化窗口的大小将根据输入的大小进行自动调整，以使输出的特征图大小为 (1, 1)。
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:  # 剩下的降采样块输出通道数保持不变
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    # 定义blk模块的前向传播函数
    Y = blk(X)  # x经过blk模块处理后输出特征图Y
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)  # 生成锚框
    cls_preds = cls_predictor(Y)  # 预测锚框类别
    bbox_preds = bbox_predictor(Y)  # 预测锚框的偏移量
    return (Y, anchors, cls_preds, bbox_preds)


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    # 一个简单的SSDnet的完整模型
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]  # 每个大模块的输入通道
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, input):
        # [None] * 5 表示一个包含5个元素的列表，每个元素都是 None
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            # 一次走完一个大模块，即提取特征，预测类别和偏移量
            output, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                input, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
            input = output
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

# test：
# net = TinySSD(num_classes=1)
# X = torch.zeros((32, 3, 256, 256))
# anchors, cls_preds, bbox_preds = net(X)
#
# print('output anchors:', anchors.shape)
# print('output class preds:', cls_preds.shape)
# print('output bbox preds:', bbox_preds.shape)


# 定义损失函数
# 关于锚框类别的损失：交叉熵损失函数
# 关于正类锚框偏移量的损失：预测偏移量是一个回归问题。使用L1范数损失，即预测值和真实值之差的绝对值。
# 掩码变量bbox_masks令负类锚框和填充锚框不参与损失的计算。
# 最后，将锚框类别和偏移量的损失相加，以获得模型的最终损失函数。
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    # 计算关于锚框类别和锚框偏移量的总损失值
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


data_dir = 'E:\CODE\Python\PyCharm\Project\learn_pytorch\practice\dataset\\banana-detection\\banana-detection'
train_dataset, test_dataset = load_data_bananas(data_dir=data_dir, batch_size=32)

myNet = TinySSD(num_classes=1)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
myNet = myNet.to(device)

myOptim = torch.optim.SGD(myNet.parameters(), lr=0.01)

writer = SummaryWriter("logs/SSDnet")
train_step = 0
def train(epoch):
    global train_step
    for data in train_dataset:
        myOptim.zero_grad()
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = myNet(imgs)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, targets)
        # 根据类别和偏移量的预测和标注值计算损失函数
        train_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        # 计算梯度需要确保train_loss是标量，如果不是，则需要进行均值等操作
        train_loss.mean().backward()
        myOptim.step()
        train_step += 1
        if train_step % 10 == 0:
            print('[epoch = {}, train_step = {}], Loss = {}'.format(epoch + 1, train_step, train_loss.mean().item()))
            writer.add_scalar("train", train_loss.mean().item(), train_step)


def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum()) / cls_labels.numel()


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum()) / bbox_labels.numel()


test_step = 0
def test(epoch):
    global test_step
    with torch.no_grad():
        for data in test_dataset:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            anchors, cls_preds, bbox_preds = myNet(imgs)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, targets)
            cls_accuracy = cls_eval(cls_preds, cls_labels)
            bbox_err = bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            test_step += 1
            writer.add_scalar("test_cls_accuracy", cls_accuracy, test_step)
            writer.add_scalar("test_bbox_err", bbox_err, test_step)


if __name__ == '__main__':
    input = torch.ones((32, 3, 256, 256))
    input = input.to(device)
    writer.add_graph(myNet, input)
    for epoch in range(10):
        train(epoch)
        test(epoch)

    writer.close()

