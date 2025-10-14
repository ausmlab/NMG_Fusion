import numpy as np
import torch

from shapely.geometry import Polygon

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


def get_gt_annos(gt_dicts):
    '''
    args:
        #txt_file: dota-style json filw where bbox format of dota style is x1, y1, x2, y2, x3, y3, x4, y4
        #class_names: list of class names in order of predcitions
        #imgs: list of image names
        #ROOT: path of annotation files
    return:
        gt_annos: dictionary of [image_id, x1, y1, x2, y2, x3, y3, x4, y4] by class
    '''
    gt_annos = {}
    for class_name in ['pylon', 'powerline']:
        gt_annos[class_name] = []

    for n, gt in enumerate(gt_dicts):
        poly_gts = obb2poly_le90(gt[:,:-1])
        clses = gt[:, -1]
        for i, poly_gt in enumerate(poly_gts):
            cls = clses[i]
            if cls == 0:
                class_name = 'pylon'
            else:
                class_name = 'powerline'
            gt_annos[class_name].append([n] + list(poly_gt))

    for class_name in ['pylon', 'powerline']:
        gt_annos[class_name] = np.array(gt_annos[class_name])

    return gt_annos

def get_sorted_preds(pred_dicts, score_th =0.5) :
    '''
    args:
        pred_dicts: dictionary of result of `inference_detector` api in mmrotate with key of image_name
        class_names: list of class names in order of predcitions
        imgs: list of image names
    retrun:
        sorted_preds: dictionary of lists of [score, [x1, y1, x2, y2, x3, y3, x4, y4], image_id] sorted by scores according to the class
    '''

    preds = {}
    sorted_preds = {}
    for class_name in ['pylon', 'powerline']  :
        preds[class_name] = []
        sorted_preds[class_name] = []

    for img_id, output  in enumerate(pred_dicts) :
        clses_per_file = output[:,-1] # 300
        scored_per_file = output[:,-2] # 300,
        preds_per_file = obb2poly_le90(output[:,:-2]) # 300, 8
        for n, pred_per_file in enumerate(preds_per_file) :
            cls = clses_per_file[n]
            score = scored_per_file[n]
            if cls == 0 :
                class_name = 'pylon'
            else :
                class_name = 'powerline'
            #score = pred_per_file[-1]
            if score > score_th :
                polygon = list(pred_per_file)
                preds[class_name].append([score] + polygon + [img_id])

    for class_name in ['pylon', 'powerline']  :
        preds[class_name] = np.array(preds[class_name])
        if len(preds[class_name]) > 0 :
            confidence = preds[class_name][:,0]
            sorted_ind = np.argsort(-confidence)
            sorted_preds[class_name] = preds[class_name][sorted_ind, :]
        else :
            sorted_preds[class_name] = []

    return sorted_preds


def compute_iou(poly1, poly2):
    """
    Compute the IoU between two polygons.

    Args:
    poly1, poly2: List of coordinates [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
    IoU value (float)
    """
    # Convert lists to shapely polygons
    polygon1 = Polygon([(poly1[i], poly1[i + 1]) for i in range(0, len(poly1), 2)])
    polygon2 = Polygon([(poly2[i], poly2[i + 1]) for i in range(0, len(poly2), 2)])

    # Compute intersection and union
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area

    # Compute IoU
    iou = intersection / union if union > 0 else 0
    return iou

def get_tp (sorted_preds, gt_annos, iou_th) :
    '''
    args
        sorted_preds: list of [score, XYs, image_id] sorted by scores
        gt_annos: list of [image_id, XYs]
    return
        tp: list of correct ones
        fp: list of wrong ones
    '''
    tp = np.zeros(sorted_preds.shape[0])
    fp = np.zeros(sorted_preds.shape[0])
    gt_annos_for_cf = np.copy(gt_annos)
    for i, sorted_pred in enumerate(sorted_preds) :
        img_id = int(sorted_pred[-1])
        pred_poly = sorted_pred[1:-1]
        #print (cx, cy)
        for j in np.where(gt_annos_for_cf[:,0] == img_id)[0] :
            # get_iou
            gt_poly = gt_annos_for_cf[j,1:]
            iou = compute_iou(pred_poly, gt_poly)
            if iou >= iou_th :
                tp[i] = 1
                gt_annos_for_cf[j] = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1])
                continue
        if not tp[i] == 1 :
            fp[i] = 1
    return tp, fp

def get_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_oriented_detection(
    gt_list,
    pred_list,
    iou_threshold=0.5,
    score_threshold=0.001
):
    """
    Evaluate oriented object detection using IoU and score thresholds.

    gt_list: List[Tensor] — each with shape (N_gt_i, 6)
    pred_list: List[Tensor] — each with shape (N_pred_i, 7)
    """
    assert len(gt_list) == len(pred_list), "GT and prediction list must have same length."
    sorted_preds = get_sorted_preds(pred_list, score_th=score_threshold)  # [score, XYs, image_id]
    gt_annos = get_gt_annos(gt_list)  # [image_id, XYs]

    aps = []
    for class_name in ['pylon', 'powerline'] :
        if len(sorted_preds[class_name]) > 0 :
            tp, fp = get_tp(sorted_preds[class_name], gt_annos[class_name], iou_threshold)
            npos = len(gt_annos[class_name])
            # compute precision recall
            fps = np.cumsum(fp)
            tps = np.cumsum(tp)
            rec = tps / float(npos)
            use_07_metric = False
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
            ap = get_ap(rec, prec, use_07_metric)
            #print(class_name, ap * 100)
        else :
            ap = 0
        aps.append(ap)

    #print ('mAP', sum(aps) / 2)

    return {
        'pylon': aps[0],
        'powerline': aps[1],
        'mAP': sum(aps) / 2
    }
