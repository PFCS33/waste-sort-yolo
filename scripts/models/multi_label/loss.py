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
        # DEBUG: 函数入口信息
        LOGGER.info(f"\n=== _forward_bce DEBUG ===")
        
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        
        LOGGER.info(f"pred_scores shape: {pred_scores.shape}")
        LOGGER.info(f"batch_size: {batch_size}")
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
        
        # DEBUG: Assigner结果
        LOGGER.info(f"fg_mask shape: {fg_mask.shape}, 前景总数: {fg_mask.sum()}")
        LOGGER.info(f"target_scores_sum: {target_scores_sum}")

        # ========== Build multi-hot targets ==========
        cls_multihot = batch["cls_multihot"].to(self.device)
        batch_idx = batch["batch_idx"].to(self.device)
        
        # DEBUG: 检查输入的multi-hot targets
        LOGGER.info(f"cls_multihot shape: {cls_multihot.shape}")
        if cls_multihot.shape[0] > 0:
            LOGGER.info(f"第一个GT的cls_multihot:")
            LOGGER.info(f"  材料级别 (0-4): {cls_multihot[0, :5]}")
            LOGGER.info(f"  对象级别 (5-18): {cls_multihot[0, 5:]}")
            LOGGER.info(f"  激活总数: 材料={cls_multihot[0, :5].sum()}, 对象={cls_multihot[0, 5:].sum()}")

        target_scores_multihot = self._build_multihot_targets(
            target_gt_idx=target_gt_idx,
            fg_mask=fg_mask,
            target_scores=target_scores,
            cls_multihot=cls_multihot,
            batch_idx=batch_idx,
            batch_size=batch_size,
        )
        
        # DEBUG: 构建后的multi-hot targets分析
        fg_total = fg_mask.sum().item()
        LOGGER.info(f"target_scores_multihot shape: {target_scores_multihot.shape}")
        LOGGER.info(f"前景anchor数量: {fg_total}")
        
        if fg_total > 0:
            # 统计材料和对象级别的激活
            material_targets = target_scores_multihot[:, :, :5].sum(dim=(0,1))  # [5]
            object_targets = target_scores_multihot[:, :, 5:].sum(dim=(0,1))    # [14]
            
            LOGGER.info(f"材料级别targets激活统计: {material_targets}")
            LOGGER.info(f"对象级别targets激活统计: {object_targets}")
            LOGGER.info(f"材料级别总激活: {material_targets.sum():.2f}")
            LOGGER.info(f"对象级别总激活: {object_targets.sum():.2f}")
            
            # 检查第一个前景anchor的targets
            first_fg_batch, first_fg_anchor = torch.where(fg_mask)
            if len(first_fg_batch) > 0:
                b, a = first_fg_batch[0].item(), first_fg_anchor[0].item()
                LOGGER.info(f"第一个前景anchor [{b},{a}] targets:")
                LOGGER.info(f"  材料级别: {target_scores_multihot[b, a, :5]}")
                LOGGER.info(f"  对象级别: {target_scores_multihot[b, a, 5:]}")
                
                # 检查对应的预测值
                pred_sigmoid = pred_scores.sigmoid()
                LOGGER.info(f"  预测概率材料: {pred_sigmoid[b, a, :5].tolist()}")
                LOGGER.info(f"  预测概率对象: {pred_sigmoid[b, a, 5:].tolist()}")

        # ========== Classification Loss (BCE with multi-hot) ==========
        bce_loss_raw = self.bce(pred_scores, target_scores_multihot.to(dtype))
        loss[1] = bce_loss_raw.sum() / target_scores_sum
        
        # DEBUG: BCE损失详细信息
        LOGGER.info(f"BCE损失计算:")
        LOGGER.info(f"  raw BCE loss shape: {bce_loss_raw.shape}")
        LOGGER.info(f"  raw BCE loss sum: {bce_loss_raw.sum().item():.6f}")
        LOGGER.info(f"  normalized BCE loss: {loss[1].item():.6f}")
        
        if fg_total > 0:
            # 分析材料和对象的BCE损失
            material_bce = bce_loss_raw[:, :, :5].sum().item()
            object_bce = bce_loss_raw[:, :, 5:].sum().item()
            LOGGER.info(f"  材料级别BCE损失: {material_bce:.6f}")
            LOGGER.info(f"  对象级别BCE损失: {object_bce:.6f}")
            LOGGER.info(f"  损失比例 (对象/材料): {object_bce/material_bce if material_bce > 0 else float('inf'):.2f}")

        # ========== Box Loss ==========
        box_loss_computed = False
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
            box_loss_computed = True

        # DEBUG: 最终损失信息
        LOGGER.info(f"损失分量 (归一化前):")
        LOGGER.info(f"  Box loss: {loss[0].item():.6f} {'(computed)' if box_loss_computed else '(not computed)'}")
        LOGGER.info(f"  Cls loss: {loss[1].item():.6f}")
        LOGGER.info(f"  DFL loss: {loss[2].item():.6f}")

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        
        final_loss = loss * batch_size
        LOGGER.info(f"损失分量 (最终):")
        LOGGER.info(f"  Box: {loss[0].item():.6f} * {batch_size} = {final_loss[0].item():.6f}")
        LOGGER.info(f"  Cls: {loss[1].item():.6f} * {batch_size} = {final_loss[1].item():.6f}")
        LOGGER.info(f"  DFL: {loss[2].item():.6f} * {batch_size} = {final_loss[2].item():.6f}")
        LOGGER.info(f"总损失: {final_loss.sum().item():.6f}")

        return final_loss, loss.detach()

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

        # DEBUG: 检查输入的multi-hot targets
        LOGGER.info(f"\n=== DEBUG Hierarchical Softmax Loss ===")
        LOGGER.info(f"cls_multihot shape: {cls_multihot.shape}")
        LOGGER.info(f"cls_multihot数据类型: {cls_multihot.dtype}")
        if cls_multihot.shape[0] > 0:
            LOGGER.info(f"第一个GT的cls_multihot:")
            LOGGER.info(f"  材料级别 (0-4): {cls_multihot[0, :5]}")
            LOGGER.info(f"  对象级别 (5-18): {cls_multihot[0, 5:]}")
            LOGGER.info(f"  激活总数: 材料={cls_multihot[0, :5].sum()}, 对象={cls_multihot[0, 5:].sum()}")
            
            # 显示前几个GT的multihot向量
            num_gts = min(3, cls_multihot.shape[0])
            for i in range(num_gts):
                material_active = torch.where(cls_multihot[i, :5] == 1)[0]
                object_active = torch.where(cls_multihot[i, 5:] == 1)[0] + 5
                LOGGER.info(f"GT[{i}]: 材料激活={material_active.tolist()}, 对象激活={object_active.tolist()}")

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

        # DEBUG: 函数入口信息
        LOGGER.info(f"\n=== _hierarchical_softmax_loss DEBUG ===")
        LOGGER.info(f"pred_scores shape: {pred_scores.shape}")
        LOGGER.info(f"fg_mask shape: {fg_mask.shape}, fg总数: {fg_mask.sum()}")
        LOGGER.info(f"batch_size: {batch_size}")
        LOGGER.info(f"object_groups: {self.object_groups}")

        total_loss = torch.tensor(0.0, device=device)
        num_fg = 0

        for b in range(batch_size):
            gt_mask = batch_idx == b
            gt_global_indices = torch.where(gt_mask)[0]
            num_gt = len(gt_global_indices)

            LOGGER.info(f"\n--- Batch {b} ---")
            LOGGER.info(f"GT数量: {num_gt}, GT indices: {gt_global_indices.tolist()}")

            if num_gt == 0:
                LOGGER.info("跳过: 没有GT")
                continue

            fg_anchor_mask = fg_mask[b]
            if not fg_anchor_mask.any():
                LOGGER.info("跳过: 没有前景anchor")
                continue

            fg_indices = torch.where(fg_anchor_mask)[0]
            num_fg_this = len(fg_indices)
            num_fg += num_fg_this
            LOGGER.info(f"前景anchor数量: {num_fg_this}")

            # Get predictions for foreground anchors
            fg_pred = pred_scores[b, fg_indices]  # [num_fg, nc_total]

            # Get matched GT info
            local_gt_idx = target_gt_idx[b, fg_indices].clamp(0, num_gt - 1)
            global_gt_idx = gt_global_indices[local_gt_idx]
            matched_multihot = cls_multihot[global_gt_idx]  # [num_fg, nc_total]

            LOGGER.info(f"local_gt_idx: {local_gt_idx.tolist()}")
            LOGGER.info(f"global_gt_idx: {global_gt_idx.tolist()}")

            # Soft weight from assigner
            soft_weight = target_scores[b, fg_indices].max(dim=-1)[0]  # [num_fg]
            LOGGER.info(f"soft_weight: {soft_weight.tolist()}")

            # ========== Level 1: Material Softmax ==========
            material_logits = fg_pred[:, : h.nc_material]  # [num_fg, 5]
            material_targets = matched_multihot[:, : h.nc_material].argmax(
                dim=-1
            )  # [num_fg]

            LOGGER.info(f"材料级别:")
            LOGGER.info(f"  logits shape: {material_logits.shape}")
            LOGGER.info(f"  targets: {material_targets.tolist()}")
            if num_fg_this > 0:
                material_probs = F.softmax(material_logits, dim=-1)
                LOGGER.info(f"  第一个anchor的材料概率: {material_probs[0].tolist()}")
                LOGGER.info(f"  第一个anchor的材料multihot: {matched_multihot[0, :h.nc_material].tolist()}")

            loss_material = F.cross_entropy(
                material_logits, material_targets, reduction="none"
            )
            loss_material = (loss_material * soft_weight).sum()
            LOGGER.info(f"  材料损失: {loss_material.item():.6f}")

            # ========== Level 2: Object Softmax (conditional p) ==========
            loss_object = torch.tensor(0.0, device=device)
            LOGGER.info(f"对象级别:")

            for mat_id, obj_indices in self.object_groups.items():
                if len(obj_indices) <= 1:
                    # Only one object in this material, no softmax needed
                    LOGGER.info(f"  材料{mat_id}: 只有1个对象，跳过softmax")
                    continue

                # Find anchors matched to this material
                mat_mask = material_targets == mat_id
                if not mat_mask.any():
                    LOGGER.info(f"  材料{mat_id}: 没有匹配的anchor")
                    continue

                num_mat_anchors = mat_mask.sum().item()
                LOGGER.info(f"  材料{mat_id}: {num_mat_anchors}个anchor, 对象indices: {obj_indices}")

                # Get object logits for this material's objects
                obj_logits = fg_pred[mat_mask][:, obj_indices]  # [num_mat, num_obj]

                # Get object targets
                obj_multihot = matched_multihot[mat_mask][
                    :, obj_indices
                ]  # [num_mat, num_obj]
                obj_targets = obj_multihot.argmax(dim=-1)  # [num_mat]

                LOGGER.info(f"    对象targets: {obj_targets.tolist()}")
                if num_mat_anchors > 0:
                    obj_probs = F.softmax(obj_logits, dim=-1)
                    LOGGER.info(f"    第一个anchor的对象概率: {obj_probs[0].tolist()}")
                    LOGGER.info(f"    第一个anchor的对象multihot: {obj_multihot[0].tolist()}")

                # Cross entropy
                loss_obj_mat = F.cross_entropy(
                    obj_logits, obj_targets, reduction="none"
                )
                loss_obj_mat = (loss_obj_mat * soft_weight[mat_mask]).sum()
                LOGGER.info(f"    材料{mat_id}对象损失: {loss_obj_mat.item():.6f}")
                loss_object = loss_object + loss_obj_mat

            LOGGER.info(f"  总对象损失: {loss_object.item():.6f}")
            batch_total_loss = loss_material + loss_object
            LOGGER.info(f"Batch {b} 总损失: {batch_total_loss.item():.6f} (材料: {loss_material.item():.6f} + 对象: {loss_object.item():.6f})")
            total_loss = total_loss + batch_total_loss

        LOGGER.info(f"\n=== 最终结果 ===")
        LOGGER.info(f"总前景anchor数: {num_fg}")
        LOGGER.info(f"总损失: {total_loss.item():.6f}")
        return total_loss

    def _forward_penalty(self, preds, batch):
        """hYOLO-style loss: BCE + hierarchy violation penalty"""
        # DEBUG: 函数入口信息
        LOGGER.info(f"\n=== _forward_penalty DEBUG ===")
        
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        
        LOGGER.info(f"pred_scores shape: {pred_scores.shape}")
        LOGGER.info(f"batch_size: {batch_size}")
        LOGGER.info(f"consistency_weight: {self.consistency_weight}")
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
        
        # DEBUG: Assigner结果
        LOGGER.info(f"fg_mask shape: {fg_mask.shape}, 前景总数: {fg_mask.sum()}")
        LOGGER.info(f"target_scores_sum: {target_scores_sum}")

        # Build multi-hot targets
        cls_multihot = batch["cls_multihot"].to(self.device)
        batch_idx = batch["batch_idx"].to(self.device)
        
        # DEBUG: 检查输入的multi-hot targets
        LOGGER.info(f"cls_multihot shape: {cls_multihot.shape}")
        if cls_multihot.shape[0] > 0:
            LOGGER.info(f"第一个GT的cls_multihot:")
            LOGGER.info(f"  材料级别 (0-4): {cls_multihot[0, :5]}")
            LOGGER.info(f"  对象级别 (5-18): {cls_multihot[0, 5:]}")

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
        
        # DEBUG: BCE损失
        LOGGER.info(f"BCE损失:")
        LOGGER.info(f"  raw BCE loss: {loss_bce.item():.6f}")
        LOGGER.info(f"  normalized BCE loss: {(loss_bce / target_scores_sum).item():.6f}")

        # ========== Hierarchy Penalty (hYOLO) ==========
        loss_penalty = self._hierarchy_penalty(
            pred_scores=pred_scores,
            fg_mask=fg_mask,
            target_scores_multihot=target_scores_multihot,
        )
        
        # DEBUG: 层级惩罚损失
        LOGGER.info(f"层级惩罚损失:")
        LOGGER.info(f"  penalty loss: {loss_penalty.item():.6f}")
        LOGGER.info(f"  weighted penalty: {(self.consistency_weight * loss_penalty).item():.6f}")

        # L_cls = L_BCE + α × L_penalty
        loss[1] = loss_bce / target_scores_sum + self.consistency_weight * loss_penalty
        
        # DEBUG: 总分类损失
        LOGGER.info(f"总分类损失: {loss[1].item():.6f} = BCE({(loss_bce / target_scores_sum).item():.6f}) + α×penalty({(self.consistency_weight * loss_penalty).item():.6f})")

        # ========== Box Loss ==========
        box_loss_computed = False
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
            box_loss_computed = True

        # DEBUG: 最终损失信息
        LOGGER.info(f"损失分量 (归一化前):")
        LOGGER.info(f"  Box loss: {loss[0].item():.6f} {'(computed)' if box_loss_computed else '(not computed)'}")
        LOGGER.info(f"  Cls loss: {loss[1].item():.6f}")
        LOGGER.info(f"  DFL loss: {loss[2].item():.6f}")

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        
        final_loss = loss * batch_size
        LOGGER.info(f"损失分量 (最终):")
        LOGGER.info(f"  Box: {loss[0].item():.6f} * {batch_size} = {final_loss[0].item():.6f}")
        LOGGER.info(f"  Cls: {loss[1].item():.6f} * {batch_size} = {final_loss[1].item():.6f}")
        LOGGER.info(f"  DFL: {loss[2].item():.6f} * {batch_size} = {final_loss[2].item():.6f}")
        LOGGER.info(f"总损失: {final_loss.sum().item():.6f}")

        return final_loss, loss.detach()

    def _hierarchy_penalty(
        self,
        pred_scores,
        fg_mask,
        target_scores_multihot,
    ):
        """
        Predicted child's parent ≠ GT parent at same anchor
        """
        # DEBUG: 层级惩罚函数入口
        LOGGER.info(f"\n--- _hierarchy_penalty DEBUG ---")
        
        h = self.h_config
        device = pred_scores.device
        obj_to_mat = self.obj_to_mat.to(device)

        batch_size = pred_scores.shape[0]
        total_penalty = torch.tensor(0.0, device=device)
        total_violations = 0
        
        LOGGER.info(f"batch_size: {batch_size}")
        LOGGER.info(f"obj_to_mat mapping: {obj_to_mat}")
        LOGGER.info(f"前景总数: {fg_mask.sum()}")

        for b in range(batch_size):
            fg_anchor_mask = fg_mask[b]
            if not fg_anchor_mask.any():
                LOGGER.info(f"Batch {b}: 跳过 (无前景anchor)")
                continue

            fg_indices = torch.where(fg_anchor_mask)[0]
            num_fg_this = len(fg_indices)
            LOGGER.info(f"Batch {b}: {num_fg_this}个前景anchor")
            
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
            
            LOGGER.info(f"  FP检测: threshold={threshold}")
            LOGGER.info(f"  预测>阈值的对象数: {(object_pred_conf > threshold).sum()}")
            LOGGER.info(f"  FP mask总数: {fp_mask.sum()}")

            if not fp_mask.any():
                LOGGER.info(f"  跳过 (无FP违反)")
                continue

            # GT material for each foreground anchor
            gt_material = fg_target[:, : h.nc_material].argmax(dim=-1)  # [num_fg]
            LOGGER.info(f"  GT材料分布: {gt_material[:min(10, len(gt_material))].tolist()}")

            # Get FP indices: (anchor_idx, obj_local_idx)
            fp_anchor_indices, fp_obj_local_indices = torch.where(fp_mask)

            if len(fp_anchor_indices) == 0:
                LOGGER.info(f"  跳过 (fp_anchor_indices为空)")
                continue

            num_fp = len(fp_anchor_indices)
            LOGGER.info(f"  FP违反数量: {num_fp}")

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
            
            # DEBUG: 详细违反信息
            violations_mask = penalty > 0
            num_violations_this = violations_mask.sum().item()
            if num_violations_this > 0:
                LOGGER.info(f"  层级违反详情:")
                LOGGER.info(f"    违反数量: {num_violations_this}/{num_fp}")
                LOGGER.info(f"    FP对象 (前5个): {fp_obj_global_indices[:5].tolist()}")
                LOGGER.info(f"    预测父材料 (前5个): {fp_parent_material[:5].tolist()}")
                LOGGER.info(f"    GT材料 (前5个): {fp_gt_material[:5].tolist()}")
                LOGGER.info(f"    delta (前5个): {delta[:5].tolist()}")
                LOGGER.info(f"    FP置信度 (前5个): {fp_conf[:5].tolist()}")
                LOGGER.info(f"    惩罚值 (前5个): {penalty[:5].tolist()}")
                LOGGER.info(f"    当前批次惩罚总和: {penalty.sum().item():.6f}")
            else:
                LOGGER.info(f"  无层级违反 (所有FP的父材料都匹配GT)")

            total_penalty = total_penalty + penalty.sum()
            total_violations += num_violations_this

        # Normalize by number of violations
        LOGGER.info(f"\n=== 层级惩罚最终结果 ===")
        LOGGER.info(f"总违反数: {total_violations}")
        LOGGER.info(f"总惩罚 (归一化前): {total_penalty.item():.6f}")
        
        if total_violations > 0:
            total_penalty = total_penalty / total_violations
            LOGGER.info(f"总惩罚 (归一化后): {total_penalty.item():.6f}")
        else:
            LOGGER.info(f"无违反，惩罚为0")

        return total_penalty

    def _build_object_to_material_tensor(self):
        """Build tensor for fast object → material lookup."""
        h = self.h_config
        self.obj_to_mat = torch.tensor(
            [h.get_material_id(i) for i in range(h.nc_total)],
            dtype=torch.long,
        )
