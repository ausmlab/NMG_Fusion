# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import numpy as np

def poly2obb_le90(polys):
    """Convert polygons to oriented bounding boxes.
    source: https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/core/bbox/transforms.py
    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    # angles = norm_angle(angles, 'le90')
    angles = (angles + np.pi / 2) % np.pi - np.pi / 2
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.
    source: https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/core/bbox/transforms.py
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()

# For 2D flips
def flip_boxes(boxes, img_w, img_h, mode='horizontal'):
    flipped = boxes.clone()
    if mode == 'horizontal':
        flipped[:, 0::2] = img_w - boxes[:, 0::2]
    elif mode == 'vertical':
        flipped[:, 1::2] = img_h - boxes[:, 1::2]
    else:
        raise ValueError("mode must be 'horizontal' or 'vertical'")
    return flipped

def reorder_clockwise(box):
    """
    Reorder a single 1x8 box to DOTA-style clockwise order starting from top-left.
    Input:  torch.tensor([x1,y1,x2,y2,x3,y3,x4,y4])
    Output: same shape, reordered
    """
    pts = box.view(4, 2)

    # 1. Find the center
    center = pts.mean(dim=0)

    # 2. Compute angles relative to center (atan2 gives CCW, so we reverse for CW)
    angles = torch.atan2(pts[:,1] - center[1], pts[:,0] - center[0])
    order = torch.argsort(angles, descending=True)  # clockwise

    pts = pts[order]

    # 3. Ensure starting point is top-left (smallest y, then x)
    top_left_idx = torch.argmin(pts[:,1] * 10000 + pts[:,0]).item()  # prioritize y first
    pts = torch.roll(pts, -top_left_idx, dims=0)

    return pts.flatten()

def reorder_all_boxes(boxes):
    return torch.stack([reorder_clockwise(b) for b in boxes])
