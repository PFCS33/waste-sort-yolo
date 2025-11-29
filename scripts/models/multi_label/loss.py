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

    def __init__(self, model, h_config, loss_type="bce", consistency_weight=35):
        super().__init__(model)
        self.h_config = h_config
        self.loss_type = loss_type
        self.consistency_weight = consistency_weight
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
        elif loss_type == "hierarchical_penalty":
            self._build_object_to_material_tensor()
        LOGGER.info("Custom Loss Loaded!")

    def __call__(self, preds, batch):
        """Compute loss based on loss_type."""

        if self.loss_type == "bce":
            LOGGER.warning(f"Calling _forward_bce")
            return self._forward_bce(preds, batch)
        elif self.loss_type == "hierarchical_softmax":
            LOGGER.warning(f"Calling _forward_hierarchical_softmax")
            return self._forward_hierarchical_softmax(preds, batch)
        elif self.loss_type == "hierarchical_penalty":
            LOGGER.warning(f"Calling _forward_penalty")
            return self._forward_penalty(preds, batch)
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

        total_loss = (pred_scores * 0).sum()
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
            loss_object = (fg_pred * 0).sum()

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

            batch_total_loss = loss_material + loss_object
            total_loss = total_loss + batch_total_loss

        return total_loss

    def _forward_penalty(self, preds, batch):
        """hYOLO-style loss: BCE + hierarchy violation penalty"""

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

        # Build multi-hot targets
        cls_multihot = batch["cls_multihot"].to(self.device)
        batch_idx = batch["batch_idx"].to(self.device)

        target_scores_multihot = self._build_multihot_targets(
            target_gt_idx,
            fg_mask,
            target_scores,
            cls_multihot,
            batch_idx,
            batch_size,
        )

        # ========== BCE Loss ==========
        loss_bce = self.bce(pred_scores, target_scores_multihot.to(dtype)).sum()

        # ========== Hierarchy Penalty (hYOLO) ==========
        loss_penalty = self._hierarchy_penalty(
            pred_scores=pred_scores,
            fg_mask=fg_mask,
            target_scores_multihot=target_scores_multihot,
        )

        # L_cls = L_BCE + α × L_penalty
        loss[1] = loss_bce / target_scores_sum + self.consistency_weight * loss_penalty

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

    def _hierarchy_penalty(
        self,
        pred_scores,
        fg_mask,
        target_scores_multihot,
    ):
        """
        Predicted child's parent ≠ GT parent at same anchor
        """
        h = self.h_config
        device = pred_scores.device
        obj_to_mat = self.obj_to_mat.to(device)

        batch_size = pred_scores.shape[0]
        total_penalty = (pred_scores * 0).sum()
        total_violations = 0

        for b in range(batch_size):
            fg_anchor_mask = fg_mask[b]
            if not fg_anchor_mask.any():
                continue

            fg_indices = torch.where(fg_anchor_mask)[0]
            num_fg_this = len(fg_indices)

            fg_pred = pred_scores[b, fg_indices]  # [num_fg, nc_total]
            fg_target = target_scores_multihot[b, fg_indices]  # [num_fg, nc_total]

            # Predicted confidences
            pred_conf = fg_pred.sigmoid()  # [num_fg, nc_total]

            # Object predictions and targets
            object_pred_conf = pred_conf[:, h.nc_material :]  # [num_fg, nc_object]
            object_target = fg_target[:, h.nc_material :]  # [num_fg, nc_object]

            # Find FALSE POSITIVES: conf > threshold AND target == 0
            threshold = 0.001
            fp_mask = (object_pred_conf > threshold) & (object_target == 0)

            if not fp_mask.any():
                continue

            # GT material for each foreground anchor
            gt_material = fg_target[:, : h.nc_material].argmax(dim=-1)  # [num_fg]
            # Get FP indices: (anchor_idx, obj_local_idx)
            fp_anchor_indices, fp_obj_local_indices = torch.where(fp_mask)

            if len(fp_anchor_indices) == 0:
                continue

            num_fp = len(fp_anchor_indices)

            # Convert local object index to global class index
            fp_obj_global_indices = fp_obj_local_indices + h.nc_material  # [num_fp]

            # Parent material for each FP object
            fp_parent_material = obj_to_mat[fp_obj_global_indices]  # [num_fp]

            # GT material at each FP anchor
            fp_gt_material = gt_material[fp_anchor_indices]  # [num_fp]

            # δ = 1 if parent matches GT, else 0
            delta = (fp_parent_material == fp_gt_material).float()  # [num_fp]

            # Confidence of each FP object
            fp_conf = object_pred_conf[
                fp_anchor_indices, fp_obj_local_indices
            ]  # [num_fp]

            # Penalty = (1 - δ) × conf
            penalty = (1 - delta) * fp_conf  # [num_fp]

            violations_mask = penalty > 0
            num_violations_this = violations_mask.sum().item()

            total_penalty = total_penalty + penalty.sum()
            total_violations += num_violations_this

        # Normalize by number of violations
        if total_violations > 0:
            total_penalty = total_penalty / total_violations

        return total_penalty

    def _build_object_to_material_tensor(self):
        """Build tensor for fast object → material lookup."""
        h = self.h_config
        self.obj_to_mat = torch.tensor(
            [h.get_material_id(i) for i in range(h.nc_total)],
            dtype=torch.long,
        )
