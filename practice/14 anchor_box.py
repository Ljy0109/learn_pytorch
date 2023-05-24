""""
编写锚框的生成程序
"""
import torch
import torchvision
import cv2

def multibox_prior(data, sizes, ratios):
    """
    生成以每个像素为中心具有不同形状的锚框
    尺寸均归一化到[0,1]
    """
    # [-2:]从倒数第二个开始往后数,[:2]是数到第二个
    in_height, in_width = data.shape[-2:]
    # num_size 缩放的种类个数，num_ratio 长宽比个数
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    # 将sizes和ratios转tensor并分配设备
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点, 中心点在[0, 1]之间
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # 参数indexing用于指定返回的网格坐标的索引方式。默认值为'ij'，表示使用行索引和列索引。
    # 也可以设置为'xy'，表示使用x轴和y轴的索引方式。
    # 以center_h为行坐标，以center_w为列坐标生成网格，shift_y表示行坐标
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，宽度和高度分别为 h*s*(r)^(1/2) 和 h*s/(r^(1/2))
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    # torch.sqrt（）开方
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # 中心-w，-h即可得到(x1, y1),+w,+h即可得到(x2,y2)
    output = out_grid + anchor_manipulations
    # unsqueeze(0) 的作用是在张量的最外层添加一个维度，
    # 可以理解为将单个样本或单个通道的数据转换为批量处理的形式，以便在深度学习模型中进行批量操作。
    return output.unsqueeze(0)

img = cv2.imread("dataset/anchor/catdog.jpg")
trans = torchvision.transforms.ToTensor()
img1 = trans(img)
h, w = img.shape[:2]

X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
boxes = Y.reshape(h, w, 5, 4)

