"""Generate multi-hot vector per label"""

import torch
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER


class HierarchicalYOLODataset(YOLODataset):
    """YOLO Dataset with hierarchical multi-label support."""

    def __init__(self, *args, h_config=None, **kwargs):
        if h_config is None:
            raise ValueError("h_config is required")

        self.h_config = h_config

        self._material_lookup = np.array(
            [h_config.get_material_id(i) for i in range(h_config.nc_total)],
            dtype=np.int32,
        )

        super().__init__(*args, **kwargs)
        LOGGER.info("Custom Dataset loaded!")

    def get_labels(self):
        """Load labels and add multi-hot cls_multihot field."""
        labels = super().get_labels()

        for label in labels:
            object_cls = label.get("cls")
            if object_cls is not None and len(object_cls) > 0:
                label["cls_multihot"] = self._get_to_multihot(object_cls)
            else:
                label["cls_multihot"] = torch.zeros(
                    (0, self.h_config.nc_total), dtype=torch.float32
                )

        return labels

    def _get_to_multihot(self, object_cls):
        object_cls_flat = np.asarray(object_cls, dtype=np.int32).flatten()
        num_gt = len(object_cls_flat)

        multi_hot = torch.zeros((num_gt, self.h_config.nc_total), dtype=torch.float32)

        gt_indices = np.arange(num_gt)

        # Object class (fine-grained, indices nc_material ~ nc_total-1)
        multi_hot[gt_indices, object_cls_flat] = 1.0

        # Material class (coarse, indices 0 ~ nc_material-1)
        material_cls = self._material_lookup[object_cls_flat]
        multi_hot[gt_indices, material_cls] = 1.0

        return multi_hot

    @staticmethod
    def collate_fn(batch: list[dict]):
        """Collate with cls_multihot support."""
        new_batch = {}
        batch = [
            dict(sorted(b.items())) for b in batch
        ]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {
                "masks",
                "keypoints",
                "bboxes",
                "cls",
                "segments",
                "obb",
                "cls_multihot",
            }:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
