"""
Generate input multi-label vector for loss calculation
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from ultralytics.utils import LOGGER
import torch.nn.functional as F


class HierarchicalDetectionLoss(v8DetectionLoss):
    """
    Detection loss with hierarchical multi-label classification.

    - Uses BCE loss with multi-hot targets (material + object)
    - Assigner still uses single-label cls for matching
    """

    def __init__(self, model, h_config, loss_type="bce"):
        super().__init__(model)
        self.h_config = h_config
        self.loss_type = loss_type
        # validate model nc matches hierarchy config
        if self.nc != h_config.nc_total:
            raise ValueError(
                f"Model nc ({self.nc}) != hierarchy nc_total ({h_config.nc_total}). "
                f"Ensure data.yaml has nc: {h_config.nc_total}"
            )
        # Precompute lookup table (for potential future use)
        self._material_lookup = torch.tensor(
            [h_config.get_material_id(i) for i in range(h_config.nc_total)],
            dtype=torch.long,
        )
        if loss_type == "hierarchical_softmax":
            self._build_hierarchy_structure()
        LOGGER.info("Custom Loss Loaded!")

    def __call__(self, preds, batch):
        """Compute loss based on loss_type."""
        if self.loss_type == "bce":
            return self._forward_bce(preds, batch)
        elif self.loss_type == "hierarchical_softmax":
            return self._forward_hierarchical_softmax(preds, batch)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _forward_bce(self, preds, batch):
        """Basic BCE with multi-label"""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode pred_bboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # ========== Assigner (uses single-label) ==========
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ========== Build multi-hot targets ==========
        cls_multihot = batch["cls_multihot"].to(self.device)
        batch_idx = batch["batch_idx"].to(self.device)

        target_scores_multihot = self._build_multihot_targets(
            target_gt_idx=target_gt_idx,
            fg_mask=fg_mask,
            target_scores=target_scores,
            cls_multihot=cls_multihot,
            batch_idx=batch_idx,
            batch_size=batch_size,
        )

        # ========== Classification Loss (BCE with multi-hot) ==========
        loss[1] = (
            self.bce(pred_scores, target_scores_multihot.to(dtype)).sum()
            / target_scores_sum
        )

        # ========== Box Loss ==========
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss * batch_size, loss.detach()

    def _build_multihot_targets(
        self,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
        cls_multihot: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Build multi-hot target scores from assigner output.

        Args:
            target_gt_idx: [batch, num_anchors]
            fg_mask: [batch, num_anchors]
            target_scores: [batch, num_anchors, nc]
            cls_multihot: [num_gt_total, nc_total]
            batch_idx: [num_gt_total]

        Returns:
            target_scores_multihot: [batch, num_anchors, nc_total]
        """
        device = target_gt_idx.device
        num_anchors = target_gt_idx.shape[1]
        nc_total = self.h_config.nc_total

        target_scores_multihot = torch.zeros(
            batch_size, num_anchors, nc_total, device=device, dtype=cls_multihot.dtype
        )

        for b in range(batch_size):
            # GTs for this image
            gt_mask = batch_idx == b
            gt_global_indices = torch.where(gt_mask)[0]
            num_gt = len(gt_global_indices)

            if num_gt == 0:
                continue

            # Foreground anchors for this image
            fg_anchor_mask = fg_mask[b]
            if not fg_anchor_mask.any():
                continue

            fg_anchor_indices = torch.where(fg_anchor_mask)[0]

            # target_gt_idx is local (0 to num_gt-1)
            local_gt_idx = target_gt_idx[b, fg_anchor_indices]
            local_gt_idx = local_gt_idx.clamp(0, num_gt - 1)

            # Convert to global index in cls_multihot
            global_gt_idx = gt_global_indices[local_gt_idx]

            # Get multi-hot for matched GTs
            matched_multihot = cls_multihot[global_gt_idx]  # [num_fg, nc_total]

            # Get IoU-based soft weight from assigner's target_scores, just take max
            soft_weight = target_scores[b, fg_anchor_indices].max(dim=-1)[0]  # [num_fg]

            # Weight the multi-hot labels
            weighted_multihot = matched_multihot * soft_weight.unsqueeze(-1)

            target_scores_multihot[b, fg_anchor_indices] = weighted_multihot

        return target_scores_multihot

    def _forward_hierarchical_softmax(self, preds, batch):
        """
        YOLO9000-style hierarchical softmax loss

        Conditional Probabilty: Loss = L_material + L_object|material
        For each GT:
        1. Softmax over materials → CE loss with material label
        2. Softmax over objects within that material → CE loss with object label
        """
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ========== Hierarchical Softmax Loss ==========
        cls_multihot = batch["cls_multihot"].to(self.device)
        batch_idx = batch["batch_idx"].to(self.device)

        loss_cls = self._hierarchical_softmax_loss(
            pred_scores=pred_scores,
            target_gt_idx=target_gt_idx,
            fg_mask=fg_mask,
            target_scores=target_scores,
            cls_multihot=cls_multihot,
            batch_idx=batch_idx,
            batch_size=batch_size,
        )

        loss[1] = loss_cls / target_scores_sum

        # ========== Box Loss ==========
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss * batch_size, loss.detach()

    def _build_hierarchy_structure(self):
        """
        e.g.
        Groups:
        - Level 0: Materials [0, 1, 2, 3, 4]
        - Level 1: Objects under each material
            - METAL: [5]
            - GLASS: [6]
            - PLASTIC: [7, 8, 9, 10, 17, 18]
            - PAPER: [11, 12, 14, 15, 16]
            - CARDBOARD: [13]
        """
        h = self.h_config
        self.material_indices = list(range(h.nc_material))
        self.object_groups = {}

        for mat_id in range(h.nc_material):
            # Find all object classes that map to this material
            obj_indices = []
            for obj_global, mat_mapped in h.object_to_material.items():
                if mat_mapped == mat_id:
                    obj_indices.append(obj_global)
            self.object_groups[mat_id] = obj_indices

    def _hierarchical_softmax_loss(
        self,
        pred_scores,
        target_gt_idx,
        fg_mask,
        target_scores,
        cls_multihot,
        batch_idx,
        batch_size,
    ):
        h = self.h_config
        device = pred_scores.device

        total_loss = torch.tensor(0.0, device=device)
        num_fg = 0

        for b in range(batch_size):
            gt_mask = batch_idx == b
            gt_global_indices = torch.where(gt_mask)[0]
            num_gt = len(gt_global_indices)

            if num_gt == 0:
                continue

            fg_anchor_mask = fg_mask[b]
            if not fg_anchor_mask.any():
                continue

            fg_indices = torch.where(fg_anchor_mask)[0]
            num_fg_this = len(fg_indices)
            num_fg += num_fg_this

            # Get predictions for foreground anchors
            fg_pred = pred_scores[b, fg_indices]  # [num_fg, nc_total]

            # Get matched GT info
            local_gt_idx = target_gt_idx[b, fg_indices].clamp(0, num_gt - 1)
            global_gt_idx = gt_global_indices[local_gt_idx]
            matched_multihot = cls_multihot[global_gt_idx]  # [num_fg, nc_total]

            # Soft weight from assigner
            soft_weight = target_scores[b, fg_indices].max(dim=-1)[0]  # [num_fg]

            # ========== Level 1: Material Softmax ==========
            material_logits = fg_pred[:, : h.nc_material]  # [num_fg, 5]
            material_targets = matched_multihot[:, : h.nc_material].argmax(
                dim=-1
            )  # [num_fg]

            loss_material = F.cross_entropy(
                material_logits, material_targets, reduction="none"
            )
            loss_material = (loss_material * soft_weight).sum()

            # ========== Level 2: Object Softmax (conditional p) ==========
            loss_object = torch.tensor(0.0, device=device)

            for mat_id, obj_indices in self.object_groups.items():
                if len(obj_indices) <= 1:
                    # Only one object in this material, no softmax needed
                    continue

                # Find anchors matched to this material
                mat_mask = material_targets == mat_id
                if not mat_mask.any():
                    continue

                # Get object logits for this material's objects
                obj_logits = fg_pred[mat_mask][:, obj_indices]  # [num_mat, num_obj]

                # Get object targets
                obj_multihot = matched_multihot[mat_mask][
                    :, obj_indices
                ]  # [num_mat, num_obj]
                obj_targets = obj_multihot.argmax(dim=-1)  # [num_mat]

                # Cross entropy
                loss_obj_mat = F.cross_entropy(
                    obj_logits, obj_targets, reduction="none"
                )
                loss_obj_mat = (loss_obj_mat * soft_weight[mat_mask]).sum()
                loss_object = loss_object + loss_obj_mat

            total_loss = total_loss + loss_material + loss_object

        return total_loss
