"""
实现根据IoU交并比，将锚框分配到真实框的功能。并得到真实的标签和偏移量
from anchor_get_target import multibox_target  # 输入需要有batch_size维度
"""

import torch


def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    # 详细注释见anchor_IoU
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))

    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    # anchors = (num_anchors, 4)
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)  # (num_anchors, num_gt_boxes)大小
    # torch.full(size, fill_value, dtype=None, device=None, requires_grad=False) 是一个函数，
    # 用于创建一个指定形状的张量，并用指定的值填充该张量的所有元素。
    # 对于每个锚框，分配的真实边界框的张量,即存储第i个锚框对应的真实框
    # (num_anchors,)与num_anchors的值没有区别，只是前者强调这是一个一维张量类型
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 对于每个锚框，寻找它与所有真实框的IoU中的最大值，并返回索引
    max_ious, indices = torch.max(jaccard, dim=1)
    # 根据阈值，决定是否分配真实边界框
    # torch.nonzero返回张量中非0元素的索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    # 锚框i在真实框j的IoU比在其他真实框中更大，且IoU大于阈值，所以锚框i必然是对应真实框j的
    anchors_bbox_map[anc_i] = box_j  # 第anc_i个锚框对应的真实框是box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        # 讨论的是所有IoU都没有超过阈值的锚框，此时就选一个最大IoU的真实框和它对应
        # 换句话说，上面操作做完之后，可能有的真实框没有对应的锚框，于是就选一个和它IoU最大的锚框对应
        # max_idx返回的是jaccard展平后的最大值索引
        max_idx = torch.argmax(jaccard)
        # 计算列数
        box_idx = (max_idx % num_gt_boxes).long()
        # 计算行数
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        # 令最大值所在的行和列均为-1，也就是说一个真实框对应一个锚框
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        # 负类框（背景）对应的偏移量参数令其为0
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框(真实框)坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)  # 将list转为tensor
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    # 返回偏移量， 掩码（过滤背景框），锚框的标签
    return (bbox_offset, bbox_mask, class_labels)


if __name__ == '__main__':
    # gt第一个元素代表类别
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                 [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]])
    # anchors_bbox_map = assign_anchor_to_bbox(ground_truth[:, 1:], anchors, device='cpu')
    bbox_offset, bbox_mask, class_labels = multibox_target(anchors.unsqueeze(dim=0),
                                                           ground_truth.unsqueeze(dim=0))
