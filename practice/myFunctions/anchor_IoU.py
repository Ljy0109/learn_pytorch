"""
计算锚框box1和真实框box2的IoU交并比
"""
import torch


def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    # 类似C++中的lambda表达式
    # box_area是函数名，boxes是输入参数，冒号后的是返回值
    # 因为(x2,y2)总是比(x1,y1)大的，(x2-x1)*(y2-y1)就是锚框面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # [:, None, :2] 的作用是：
    # [:, ...]：表示对第一维的所有元素进行索引，相当于保留原始的维度大小。
    # None：在当前位置插入一个新的维度，相当于在第一维的位置插入一个长度为 1 的维度。
    # :2：表示对第二维的前两个元素进行索引，相当于保留原始的维度大小。
    # 因为boxes2和boxes1的个数一般是不相等的，并且boxes1要与boxes2的每个元素比较，计算交点，(x1,y1)找最大的即为交点
    # 如果不加None，则只是一对一比较，增维之后就可以每个元素与每个元素比较，这是常见操作，（why？）->广播机制
    """
    boxes1[:, None, :2] 的形状为 (N, 1, 2)，boxes2[:, :2] 的形状为 (M, 2)，其中 N 是 boxes1 的行数，M 是 boxes2 的行数。
    在比较时，广播机制会自动将 boxes1[:, None, :2] 扩展为形状为 (N, M, 2) 的张量，使其与 boxes2[:, :2] 具有相同的形状。
    然后，对应位置的元素进行逐元素的最大值比较，返回一个形状为 (N, M, 2) 的张量，其中的每个元素都是对应位置上的最大值。
    """
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # inter_upperlefts & inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)，表示第i个锚框与第j个真实框的交点
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # inters为交集的高和宽，.clamp(min=0)表示对每个元素进行截断操作，将小于 0 的值设置为 0。
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areas & union_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

if __name__ == '__main__':
    box1 = torch.tensor(torch.rand(4, 4))
    box2 = torch.tensor(torch.rand(4, 4))
    IoU = box_iou(box1, box2)