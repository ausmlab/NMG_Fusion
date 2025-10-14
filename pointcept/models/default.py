import torch
import torch.nn as nn
import torch_scatter
import torch.distributed as dist
import torch.nn.functional as F

import copy

from pointcept.models.losses import build_criteria
from pointcept.models.losses import build_matcher
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.utils import sigmoid_focal_loss
from pointcept.models.losses.box_loss import kld_loss
from .builder import MODELS, build_model

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        weight_3d = 1.0,  ## <---- to be an arg.
        weight_2d = 0.1,  ## <------ to be an arg
        num_classes2d = 2,
        aux_loss = True,
        set_cost_class = 2.0,
        set_cost_bbox = 5.0,
        set_cost_giou = 2.0,
        cls_loss_coef = 5.0,  # 1.0
        bbox_loss_coef = 5.0,
        giou_loss_coef = 2.0,
        interm_loss_coef = 1.0,
        no_interm_box_loss = False,
        focal_alpha = 0.25,
        dec_layers = 3,
        use_dn = True,
        two_stage_type = 'standard',
        num_select = 300,
        nms_iou_threshold = -1,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)

        self.criteria = build_criteria(criteria)

        ##########################################
        self.weight_3d = weight_3d
        self.weight_2d = weight_2d

        # For 2D
        self.num_classes2d = num_classes2d
        self.aux_loss = aux_loss
        self.set_cost_class = set_cost_class
        self.set_cost_bbox = set_cost_bbox
        self.set_cost_giou = set_cost_giou
        self.cls_loss_coef = cls_loss_coef # 1.0
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.interm_loss_coef = interm_loss_coef
        self.no_interm_box_loss = no_interm_box_loss
        self.focal_alpha = focal_alpha
        self.dec_layers = dec_layers
        self.use_dn = use_dn
        self.two_stage_type = two_stage_type
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

        matcher = build_matcher(self.set_cost_class, self.set_cost_bbox, self.set_cost_giou, self.focal_alpha)

        # prepare weight dict
        weight_dict = {'loss_ce': self.cls_loss_coef, 'loss_bbox': self.bbox_loss_coef}  # 1.0, 5.0
        weight_dict['loss_giou'] = self.giou_loss_coef  # 2.0
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

        # for DN training
        if self.use_dn:
            weight_dict['loss_ce_dn'] = self.cls_loss_coef  # 1.0
            weight_dict['loss_bbox_dn'] = self.bbox_loss_coef  # 5.0
            weight_dict['loss_giou_dn'] = self.giou_loss_coef  # 2.0

        clean_weight_dict = copy.deepcopy(weight_dict)

        # TODO this is a hack
        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(self.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if self.two_stage_type != 'no':
            interm_weight_dict = {}
            try:
                no_interm_box_loss = self.no_interm_box_loss  # False
            except:
                no_interm_box_loss = False
            _coeff_weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
                'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            }
            try:
                interm_loss_coef = self.interm_loss_coef  # 1.0
            except:
                interm_loss_coef = 1.0
            interm_weight_dict.update(
                {k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in
                 clean_weight_dict_wo_dn.items()})
            weight_dict.update(interm_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']

        self.criteria2d = SetCriterion(self.num_classes2d, matcher=matcher, weight_dict=weight_dict,
                                 focal_alpha=self.focal_alpha, losses=losses,
                                 )  # Get Losses
        #self.criteria2d.to(device)
        #self.postprocessors = {'bbox': PostProcess_OBB(num_select=self.num_select, nms_iou_threshold=self.nms_iou_threshold)}


    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point, self.weight_3d, self.weight_2d)

        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point

        if not self.weight_3d == 0 :
            seg_logits = self.seg_head(feat)

        # For 2D
        if not self.weight_2d == 0 :
            outputs = point['out_2d']
            targets = point['target']

        # train
        if self.training:
            loss_3d = loss_2d = torch.tensor(0).to(feat.device)
            if not self.weight_3d == 0 :
                loss_3d = self.criteria(seg_logits, input_dict["segment"])
            # loss_2d
            if not self.weight_2d == 0:
                #if torch.isnan(outputs['pred_logits']).any() or torch.isinf(outputs['pred_logits']).any():
                #    print("NaN/Inf in")
                loss_dict = self.criteria2d(outputs, targets)
                weight_dict = self.criteria2d.weight_dict
                loss_2d = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            loss = self.weight_3d * loss_3d  + self.weight_2d * loss_2d  ###<--------------
            return dict(loss=loss, loss3d=loss_3d, loss2d=loss_2d)
        # eval
        elif "segment" in input_dict.keys():
            results = {}
            loss_3d = loss_2d = 0
            if not self.weight_3d == 0:
                loss_3d = self.criteria(seg_logits, input_dict["segment"])
                results['seg_logits'] = seg_logits
                results['loss3d'] = loss_3d
            if not self.weight_2d == 0:
                loss_dict = self.criteria2d(outputs, targets)
                weight_dict = self.criteria2d.weight_dict
                loss_2d = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                results['det_out'] = outputs
                results['loss2d'] = loss_2d
            loss = self.weight_3d * loss_3d + self.weight_2d * loss_2d  ###<--------------
            # post processing
            results['loss'] = loss
            return results
            #return dict(loss=loss, seg_logits=seg_logits, det_out=outputs, loss3d=loss_3d, loss2d=loss_2d)
        # test
        else:
            results = {}
            if not self.weight_3d == 0:
                results['seg_logits'] = seg_logits
            if not self.weight_2d == 0 :
                results['det_out'] = outputs
            return results

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #    box_ops.box_cxcywh_to_xyxy(src_boxes),
        #    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        # loss_giou = 1 - torch.diag(box_ops.generalized_obb_iou(src_boxes, target_boxes)) ###<----------- N x N
        # decoding boxes
        angle_denorm_fct = torch.tensor([[0., 0., 0., 0., 0.5]]).to(target_boxes.device)
        decoded_src_boxes = src_boxes - angle_denorm_fct
        decoded_target_boxes = target_boxes - angle_denorm_fct

        img_h, img_w = targets[0]['size']
        theta = torch.tensor(torch.pi).to(img_h.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h, theta], dim=0)
        decoded_src_boxes = decoded_src_boxes * scale_fct[None, :]
        decoded_target_boxes = decoded_target_boxes * scale_fct[None, :]

        # loss_giou = 1- diff_iou_rotated_2d(src_boxes[None, :, :], target_boxes[None, :, :]).squeeze()
        # loss_giou = 1 - diff_iou_rotated_2d(decoded_src_boxes[None, :, :], decoded_target_boxes[None, :, :]).squeeze()
        loss_giou = kld_loss(decoded_src_boxes[None, :, :], decoded_target_boxes[None, :, :]).squeeze()
        #loss_giou = torch.nan_to_num(loss_giou, nan=1.0)
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:4].sum() / num_boxes
            losses['loss_theta'] = loss_bbox[..., 4:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        # print (100*'#')
        # print (targets)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar,
                                                    **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


class PostProcess_OBB(nn.Module):
    """ This module converts the model's output into the format expected by the dota api"""

    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select  # 300
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]  # // num_categories (2)
        labels = topk_indexes % out_logits.shape[2]

        boxes = out_bbox  # normalized cx,cy,w,h,theta
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 5))  ###<----------

        # and from relative [0, 1] to absolute [0, height] coordinates
        angle_denorm_fct = torch.tensor([[0., 0., 0., 0., 0.5]]).to(boxes.device)
        boxes = boxes - angle_denorm_fct[:, None, :]

        img_h, img_w = target_sizes.unbind(1)
        theta = torch.tensor([torch.pi]).to(img_h.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h, theta], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

