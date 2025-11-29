import sys
import time
import torch

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils import LOGGER
import torchvision


def hierarchical_nms(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
    # hierarchical parameters
    nc_material: int = 5,
    nc_total: int = 19,
    q_critical: float = None,
):
    """
    Hierarchical NMS with Dixon's Q test (Top1) for distinctiveness.
    """
    assert 0 <= conf_thres <= 1, f"Invalid conf_thres {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid iou_thres {iou_thres}"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    device = prediction.device
    mps = "mps" in device.type
    if mps:
        prediction = prediction.cpu()

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nc_object = nc_total - nc_material
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc

    if q_critical is None:
        q_critical = DIXON_Q_CRITICAL_95.get(nc_object, 0.478)

    xc = prediction[:, 4:mi].amax(1) > conf_thres
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[
        ..., None
    ]

    time_limit = 0.5 + max_time_img * bs

    prediction = prediction.transpose(-1, -2)

    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs

    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        filt = xc[xi]
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, extra), 1)

        # ========== Hierarchical Logic ==========
        material_scores = cls[:, :nc_material]
        object_scores = cls[:, nc_material:nc_total]

        material_conf, material_cls = material_scores.max(dim=1)
        object_conf, object_cls, is_distinctive = _dixon_q_test(
            object_scores, conf_thres, q_critical
        )
        object_cls_global = object_cls + nc_material

        use_object = is_distinctive
        final_cls = torch.where(use_object, object_cls_global, material_cls)
        final_conf = torch.where(use_object, object_conf, material_conf)

        conf_mask = final_conf > conf_thres
        if not conf_mask.any():
            continue

        box = box[conf_mask]
        final_conf = final_conf[conf_mask]
        final_cls = final_cls[conf_mask]
        mask = mask[conf_mask]
        if return_idxs:
            xk = xk[conf_mask]

        x = torch.cat(
            [box, final_conf.unsqueeze(1), final_cls.unsqueeze(1).float(), mask], dim=1
        )

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # ========== NMS ==========
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes = x[:, :4] + c
        scores = x[:, 4]

        # if rotated:
        #     boxes_rot = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
        #     i = TorchNMS.fast_nms(boxes_rot, scores, iou_thres, iou_func=batch_probiou)
        # else:
        #     if "torchvision" in sys.modules:
        #         import torchvision

        #         i = torchvision.ops.nms(boxes, scores, iou_thres)
        #     else:
        #         i = TorchNMS.nms(boxes, scores, iou_thres)
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        i = i[:max_det]
        output[xi] = x[i]

        if return_idxs:
            keepi[xi] = xk[i].view(-1)

        if mps:
            output[xi] = output[xi].to(device)

        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output


def _dixon_q_test(
    object_scores, conf_threshold=0.25, q_critical=0.478  # 95% confidence
):
    """Dixon's Q test for top1"""
    num_boxes = object_scores.shape[0]
    device = object_scores.device
    if num_boxes == 0:
        return (
            torch.zeros(0, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.bool, device=device),
        )
    sorted_scores, sorted_indices = object_scores.sort(dim=1, descending=True)

    top1 = sorted_scores[:, 0]
    top2 = sorted_scores[:, 1]
    bottom = sorted_scores[:, -1]
    top1_cls = sorted_indices[:, 0]

    q_stat = (top1 - top2) / (top1 - bottom + 1e-6)

    is_distinctive = (top1 >= conf_threshold) & (q_stat >= q_critical)

    return top1, top1_cls, is_distinctive
