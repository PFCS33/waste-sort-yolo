"""
Combine custom dataset & loss
"""

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.cfg import DEFAULT_CFG

from .config import HierarchyConfig
from .dataset import HierarchicalYOLODataset
from .loss import HierarchicalDetectionLoss


class HierarchicalDetectionTrainer(DetectionTrainer):
    """Trainer for hierarchical multi-label detection."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides) if overrides else {}
        h_config_path = overrides.pop("hierarchy_config", "hierarchy_config.yaml")
        self.h_config = HierarchyConfig(h_config_path)
        LOGGER.info("Custom Trainer loaded!")
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build multi-lable dataset."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)

        return HierarchicalYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or (mode == "val"),
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            h_config=self.h_config,
        )

    def _setup_train(self):
        """Setup training with multi-lable loss."""
        super()._setup_train()
        self.criterion = HierarchicalDetectionLoss(self.model, self.h_config)
