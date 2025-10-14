# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch, os
from torch import nn
from scipy.optimize import linear_sum_assignment
from .box_loss import kld_loss_for_assigning


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class # 2.0
        self.cost_bbox = cost_bbox # 5.0
        self.cost_giou = cost_giou # 2.0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha # 0.25

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 5] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 5] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 5]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # out_bbox: 900, 5 / tgt_bbox: N, 5
            
        # Compute the giou cost betwen oriented bounding boxes
        angle_denorm_fct = torch.tensor([[0., 0., 0., 0., 0.5]]).to(tgt_bbox.device)
        denorm_out_bbox = out_bbox - angle_denorm_fct
        denorm_tgt_bbox = tgt_bbox - angle_denorm_fct

        tgt_start = 0
        for i in range(bs) :
            out_start = i*num_queries
            out_end = (i+1)*num_queries

            tgt_end = tgt_start + targets[i]['boxes'].shape[0]
            img_h, img_w = targets[i]['size']
            theta = torch.tensor([torch.pi]).to(img_h.device)
            scale_fct = torch.stack([img_w[None], img_h[None], img_w[None], img_h[None], theta], dim=1)

            denorm_out_bbox[out_start:out_end] = denorm_out_bbox[out_start:out_end] * scale_fct
            denorm_tgt_bbox[tgt_start:tgt_end] = denorm_tgt_bbox[tgt_start:tgt_end] * scale_fct
            tgt_start = tgt_end

        cost_giou = kld_loss_for_assigning(denorm_out_bbox, denorm_tgt_bbox)
        #cost_giou = torch.nan_to_num(cost_giou, nan=1.0)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu() # 2, 600, 12

        C = torch.where(torch.isfinite(C), C, torch.full_like(C, 1e6))

        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(set_cost_class, set_cost_bbox, set_cost_giou, focal_alpha):
    return HungarianMatcher(
            cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou,
            focal_alpha=focal_alpha
        )