"""
hierarchical/predictor.py

Hierarchical Detection Predictor with Dixon's Q fallback.
"""

import torch
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops

from .config import HierarchyConfig
from .nms import hierarchical_nms


class HierarchicalDetectionPredictor(DetectionPredictor):
    """Predictor with hierarchical NMS"""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

        # Load hierarchy config
        h_config_path = (overrides or {}).get(
            "hierarchy_config", "hierarchy_config.yaml"
        )
        self.h_config = HierarchyConfig(h_config_path)
        self.q_critical = (overrides or {}).get("q_critical", None)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        # If no hierarchy config, use standard postprocess
        if self.h_config is None:
            return super().postprocess(preds, img, orig_imgs, **kwargs)
        save_feats = getattr(self, "_feats", None) is not None
        preds = hierarchical_nms(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            classes=self.args.classes,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
            # hierarchical parameters
            nc_material=self.h_config.nc_material,
            nc_total=self.h_config.nc_total,
            q_critical=self.q_critical,
        )

        # Convert orig_imgs if needed
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f

        return results
