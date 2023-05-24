"""
画出指定坐标的所有锚框
https://blog.csdn.net/weixin_44604887/article/details/113046955
"""

import numpy as np  # 可能用到的数据值计算库
import os           # 可能用到的文件操作
import matplotlib.pyplot as plt   		# 图形绘制
import matplotlib.patches as patches 	# 添加矩形框
import matplotlib.image as image  		# 读取图像数据
import torch
import torchvision
import cv2
from myFunctions.anchor_box import multibox_prior


def draw_rectangle(bbox=[], mode=True, color='k', fill=False):
    '''绘制矩形框
        bbox：边界框数据（默认框数据不超过图片边界）
        mode: 边界框数据表示的模式
             True:  to (x1,y1,x2,y2)
             False: to (x,y,w,h)
        color: 边框颜色
        fill: 是否填充
    '''
    if mode is True:  # to (x1,y1,x2,y2)
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0] + 1  # 考虑到实际长度由像素个数决定，因此加1（可按坐标轴上两点间的点数推导）
        h = bbox[3] - bbox[1] + 1
    else:  # to (x,y,w,h)
        # 默认绘制的框不超出边界
        x = bbox[0] - bbox[2] / 2.0
        y = bbox[1] - bbox[3] / 2.0
        w = bbox[2]
        h = bbox[3]

    # 绘制边界框
    # patches.Rectangle需要传入左上角坐标、矩形区域的宽度、高度等参数
    # 获取绘制好的图形的返回句柄——用于添加到当前的图像窗口中
    rect = patches.Rectangle((x, y), w, h,
                             linewidth=1,  # 线条宽度
                             edgecolor=color,  # 线条颜色
                             facecolor='y',  #
                             fill=fill, linestyle='-')

    return rect


def draw_anchor(ax, boxes, img_height, img_width, color='r'):
    '''绘制锚框————同一中心点三个不同大小的锚框
        ax: plt的窗体句柄——用于调用矩形绘制
        center：中心点坐标
        length：基本长度
        scales：尺寸
        ratios：长宽比
        img_height: 图片高
        img_width: 图片宽

        一个锚框的大小，由基本长度+尺寸+长宽比有关
        同时锚框的最终计算值与图片实际大小有关——不能超过图片实际范围嘛
    '''

    bboxs = []  # 这里的边界框bbox是指的锚框

    # for scale in scales:  # 遍历尺寸情况
    #     for ratio in ratios:  # 同一尺寸下遍历不同的长宽比情况
    #         # 利用基本长度、尺寸与长宽比进行锚框长宽的转换
    #         # h = length * scale * np.math.sqrt(ratio)
    #         # w = length * scale / np.math.sqrt(ratio)
    #         # 利用求得的长宽，确定绘制矩形需要的左上角顶点坐标和右下角顶点坐标
    #         # 不同的绘制API可能有不同的参数需要，相应转换即可
    #         x1 = max(center[0] - w / 2., 0.)  # 考虑边界问题
    #         y1 = max(center[1] - h / 2., 0.)
    #         x2 = min(center[0] + w / 2. - 1.0, img_width - 1.)  # center[0] + w / 2 -1.0 是考虑到边框不超过边界
    #         y2 = min(center[1] + h / 2. - 1.0, img_height - 1.)
    #
    #         bbox = [x1, y1, x2, y2]
    #         print('An Anchor: ', bbox)
    #         bboxs.append(bbox)  # 押入生成的anchor

    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(x1, 0.)  # 考虑边界问题
        y1 = max(y1, 0.)
        x2 = min(x2, img_width - 1.)  # center[0] + w / 2 -1.0 是考虑到边框不超过边界
        y2 = min(y2, img_height - 1.)
        bbox = [x1, y1, x2, y2]

        # 绘制anchor的矩形框
        rect = draw_rectangle(bbox, mode=True, color=color)
        ax.add_patch(rect)


# 先读取图像，再绘制
# fig = plt.figure(figsize=(12, 8))
plt.figure(1)
ax = plt.gca()

# 图片路径
img = cv2.imread("dataset/anchor/catdog.jpg")
h, w = img.shape[:2]

b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])

X = torch.rand(size=(1, 3, h, w))
bboxes = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
bboxes = bboxes.reshape(h, w, 5, 4)
center = [620, 710]
bbox_scale = torch.tensor((w, h, w, h))
boxes = np.array(bboxes[center[0], center[1], :, :] * bbox_scale)

# img_path = os.path.join(os.getcwd(), 'img', '1.jpg')
# img = image.imread(img_path) # 读取图片数据
plt.imshow(img)  # 展示图片
print(img.shape[0])
print(img.shape[1])

draw_anchor(ax=ax, boxes=boxes,
			img_height=h, img_width=w,
			color='r')

plt.show()





