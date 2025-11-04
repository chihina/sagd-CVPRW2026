from typing import List, Tuple

import sys
import numpy as np
import torch
import torchmetrics as tm

# import torch.nn.functional as F
import torchvision.transforms.functional as TF

# import torchmetrics.functional as MF
from torchmetrics import AveragePrecision

# from torchmetrics.classification.auroc import BinaryAUROC
from torchmetrics.functional.classification.auroc import binary_auroc

from src.utils import generate_binary_gaze_heatmap, generate_gaze_heatmap, spatial_argmax2d

from torch_linear_assignment import batch_linear_assignment, assignment_to_indices


class Distance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_point_pred: torch.Tensor,
        gaze_point_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        mask = inout_gt == 1
        if mask.any():
            self.sum_dist += (gaze_point_gt[mask] - gaze_point_pred[mask]).pow(2).sum(1).sqrt().sum()
            self.n_observations += mask.sum()

    def compute(self):
        if self.n_observations != 0:
            dist = self.sum_dist / self.n_observations  # type: ignore
        else:
            dist = torch.tensor(-1000.0, device=self.device)
        return dist


class GFTestDistance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist_to_avg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_avg_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_min_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, gaze_point_pred: torch.Tensor, gaze_point_gt: torch.Tensor):
        for k, (gp_pred, gp_gt) in enumerate(zip(gaze_point_pred, gaze_point_gt)):
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points

            # Compute average gaze point
            gp_gt_avg = gp_gt.mean(0)
            # Compute distance from pred to avg gt point
            self.sum_dist_to_avg += (gp_gt_avg - gp_pred).pow(2).sum().sqrt()
            # Compute avg distance between pred and gt points
            self.sum_avg_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().mean()
            # Compute min distance between pred and gt points
            self.sum_min_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().min()
        self.n_observations += len(gaze_point_pred)

    def compute(self):
        dist_to_avg = self.sum_dist_to_avg / self.n_observations
        avg_dist = self.sum_avg_dist / self.n_observations
        min_dist = self.sum_min_dist / self.n_observations
        return dist_to_avg, avg_dist, min_dist


class AUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        """
        Computes AUC for general datasets.
        """
        super().__init__()
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_heatmap_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        for hm_pred, hm_gt, io_gt in zip(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt):
            if io_gt == 1:
                hm_gt_binary = (hm_gt > 0).int()
                self.sum_auc += binary_auroc(hm_pred, hm_gt_binary)
        self.n_observations += (inout_gt==1).sum()

    def compute(self):
        if self.n_observations != 0:
            auc = self.sum_auc / self.n_observations
        else:
            auc = torch.tensor(-1000.0, device=self.device)
        return auc


class GFTestAUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        """
        Computes AUC for GazeFollow Test set. The AUC is computed for each image in the batch, after resizing the predicted
        heatmap to the original size of the image. The ground-truth binary heatmap is generated from the ground-truth gaze
        point(s) in the original image size. At the end, the mean is returned.
        """

        super().__init__()
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_pt: torch.Tensor,
    ):
        size = gaze_heatmap_pred.shape[1:]  # (b, h, w) >> (h, w)
        for hm_pred, gp_gt in zip(gaze_heatmap_pred, gaze_pt):
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points
            hm_gt_binary = generate_binary_gaze_heatmap(gp_gt, size=size)
            self.sum_auc += binary_auroc(hm_pred, hm_gt_binary)
        self.n_observations += len(gaze_heatmap_pred)

    def compute(self):
        auc = self.sum_auc / self.n_observations
        return auc

# class GroupIoU():
#     def __init__(self):
#         pass
    
#     def gen_matrix(self, grouping_pred: torch.Tensor, grouping_gt: torch.Tensor):
#         self.grouping_pred = grouping_pred
#         self.grouping_gt = grouping_gt

#         batch_size, token_size, _ = grouping_pred.shape
#         group_ious = torch.zeros(batch_size, token_size, token_size, device=grouping_gt.device)
#         for b_idx in range(batch_size):
#             for i in range(token_size):
#                 for j in range(token_size):
#                     gt = grouping_gt[b_idx, i]
#                     pred = grouping_pred[b_idx, j]

#                     # if values are matched (both 0 or both 1) then intersection else union
#                     intersection = torch.sum((gt==1) & (pred==1)).float()
#                     union = torch.sum((gt==1) | (pred==1)).float()
#                     if union==0:
#                         # iou = torch.tensor(1.0, device=grouping_gt.device)
#                         iou = torch.tensor(0.0, device=grouping_gt.device)  # if both empty, iou=0
#                     else:
#                         iou = intersection / union
                    
#                     group_ious[b_idx, i, j] = iou

#         self.group_ious = group_ious

#     def match(self):
#         # find best matching using hungarian algorithm
#         assignment = batch_linear_assignment(-self.group_ious)
#         self.row_ind, self.col_ind = assignment_to_indices(assignment)

#     def compute(self):
#         group_ious_opt = self.group_ious[torch.arange(self.group_ious.shape[0]).unsqueeze(1), self.row_ind, self.col_ind]
#         mask = torch.any(torch.sum(self.grouping_gt, dim=-1, keepdim=True)!=0, dim=-1)
#         group_ious_opt_masked = group_ious_opt[mask]
#         grouping_pred_opt_masked = self.grouping_pred[torch.arange(self.grouping_pred.shape[0]).unsqueeze(1), self.col_ind][mask]
#         grouping_gt_masked = self.grouping_gt[mask]

#         return group_ious_opt, group_ious_opt_masked, grouping_pred_opt_masked, grouping_gt_masked

#     def compute_hm_dist(self, hm_pred, hm_gt):
#         hm_pred_opt = hm_pred[torch.arange(hm_pred.shape[0]).unsqueeze(1), self.col_ind]
#         hm_gt_opt = hm_gt[torch.arange(hm_gt.shape[0]).unsqueeze(1), self.row_ind]
#         batch_size, token_size, hm_h, hm_w = hm_pred_opt.shape
#         hm_pt_pred = spatial_argmax2d(hm_pred_opt.reshape(-1, hm_h, hm_w), normalize=True).view(batch_size, token_size, -1)
#         hm_pt_gt = spatial_argmax2d(hm_gt_opt.reshape(-1, hm_h, hm_w), normalize=True).view(batch_size, token_size, -1)

#         hm_dist = (hm_pt_gt - hm_pt_pred).pow(2).sum(-1).sqrt()  # (b, token_size)
#         mask = torch.any(torch.sum(self.grouping_gt, dim=-1, keepdim=True)!=0, dim=-1)  # (b, token_size)
#         hm_dist_masked = hm_dist[mask]

#         return hm_dist, hm_dist_masked


class GroupIoU():
    def __init__(self):
        pass

    def gen_matrix(self, grouping_pred: torch.Tensor, grouping_gt: torch.Tensor):
        """
        A vectorized version of the IoU matrix generation that avoids Python for-loops
        for significant speedup.
        """
        self.grouping_pred = grouping_pred
        self.grouping_gt = grouping_gt

        # Get shapes
        # grouping_gt shape: (B, N, P)  (Batch, Num_GT_Groups, Num_People)
        # grouping_pred shape: (B, M, P) (Batch, Num_Pred_Groups, Num_People)
        B, N, P = grouping_gt.shape
        M = grouping_pred.shape[1]

        # Expand dimensions for broadcasting
        # gt becomes (B, N, 1, P), pred becomes (B, 1, M, P)
        gt_expanded = grouping_gt.unsqueeze(2)
        pred_expanded = grouping_pred.unsqueeze(1)
        
        # --- Vectorized IoU Calculation ---
        # The '&' and '|' operations will be broadcast, resulting in a (B, N, M, P) tensor
        intersection_matrix = (gt_expanded == 1) & (pred_expanded == 1)
        union_matrix = (gt_expanded == 1) | (pred_expanded == 1)

        # Sum over the last dimension (P) to get counts for each group pair
        # The result of both sums will be a (B, N, M) tensor
        intersection_counts = intersection_matrix.sum(dim=-1).float()
        union_counts = union_matrix.sum(dim=-1).float()

        # Divide to get the IoU. Use nan_to_num to handle the union_counts == 0 case.
        # This will set the result to 0.0 where division by zero occurs.
        group_ious = torch.nan_to_num(intersection_counts / union_counts, nan=0.0)

        # In your original code, N and M are both 'token_size', so the shape is (B, N, N)
        self.group_ious = group_ious

    def match(self):
        # find best matching using hungarian algorithm
        # This part remains the same
        assignment = batch_linear_assignment(-self.group_ious)
        self.row_ind, self.col_ind = assignment_to_indices(assignment)

    def compute(self):
        # This part remains the same
        group_ious_opt = self.group_ious[torch.arange(self.group_ious.shape[0]).unsqueeze(1), self.row_ind, self.col_ind]
        mask = torch.any(torch.sum(self.grouping_gt, dim=-1, keepdim=True) != 0, dim=-1)
        group_ious_opt_masked = group_ious_opt[mask]
        grouping_pred_opt_masked = self.grouping_pred[torch.arange(self.grouping_pred.shape[0]).unsqueeze(1), self.col_ind][mask]
        grouping_gt_masked = self.grouping_gt[mask]

        return group_ious_opt, group_ious_opt_masked, grouping_pred_opt_masked, grouping_gt_masked

    def compute_hm_dist(self, hm_pred, hm_gt):
        # This part remains the same
        hm_pred_opt = hm_pred[torch.arange(hm_pred.shape[0]).unsqueeze(1), self.col_ind]
        hm_gt_opt = hm_gt[torch.arange(hm_gt.shape[0]).unsqueeze(1), self.row_ind]
        batch_size, token_size, hm_h, hm_w = hm_pred_opt.shape
        hm_pt_pred = spatial_argmax2d(hm_pred_opt.reshape(-1, hm_h, hm_w), normalize=True).view(batch_size, token_size, -1)
        hm_pt_gt = spatial_argmax2d(hm_gt_opt.reshape(-1, hm_h, hm_w), normalize=True).view(batch_size, token_size, -1)

        hm_dist = (hm_pt_gt - hm_pt_pred).pow(2).sum(-1).sqrt()  # (b, token_size)
        mask = torch.any(torch.sum(self.grouping_gt, dim=-1, keepdim=True) != 0, dim=-1)  # (b, token_size)
        hm_dist_masked = hm_dist[mask]

        return hm_dist, hm_dist_masked

class GroupAP():
    # def __init__(self, iou_thresh=0.5):
    def __init__(self, iou_thresh=0.75):
        self.iou_thresh = iou_thresh

    def calculateAveragePrecision(self, rec, prec):
        mrec = [0] + [e for e in rec] + [1]
        mpre = [0] + [e for e in prec] + [0]

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ii = []

        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)

        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    def compute(self, grouping_pred: torch.Tensor, grouping_gt: torch.Tensor, hm_pred: torch.Tensor, hm_gt: torch.Tensor):
        TP = []
        FP = []
        dists = []
        npos = 0
        npos_dist = 0
        for b in range(grouping_gt.shape[0]):
            grouping_gt_b = grouping_gt[b]
            grouping_gt_b = grouping_gt_b[grouping_gt_b.sum(dim=-1)>1]  # remove empty groups
            npos += grouping_gt_b.shape[0]

            grouping_pred_b = grouping_pred[b]
            grouping_pred_b = grouping_pred_b[grouping_pred_b.sum(dim=-1)>1]  # remove empty groups

            det_gt = torch.zeros(grouping_gt_b.shape[0], dtype=torch.bool, device=grouping_gt.device)

            for pred_idx in range(grouping_pred_b.shape[0]):
                pred = grouping_pred_b[pred_idx]
                ious = []
                for gt_idx in range(grouping_gt_b.shape[0]):
                    gt = grouping_gt_b[gt_idx]
                    intersection = torch.sum((gt==1) & (pred==1)).float()
                    union = torch.sum((gt==1) | (pred==1)).float()
                    if union==0:
                        iou = torch.tensor(0.0, device=grouping_gt.device)  # if both empty, iou=0
                    else:
                        iou = intersection / union
                    ious.append(iou.item())
                ious = torch.tensor(ious)
                max_iou = ious.max() if len(ious)>0 else torch.tensor(0.0)

                if max_iou >= self.iou_thresh:
                    if not det_gt[ious.argmax()]:
                        TP.append(1)
                        FP.append(0)

                        hm_pred_target = hm_pred[b, pred_idx]
                        hm_pred_target_pnt = spatial_argmax2d(hm_pred_target, normalize=True)
                        hm_gt_target = hm_gt[b, ious.argmax()]
                        hm_gt_target_pnt = spatial_argmax2d(hm_gt_target, normalize=True)
                        hm_dist = (hm_gt_target_pnt - hm_pred_target_pnt).pow(2).sum(-1).sqrt()  # (b, token_size)
                        dists.append(hm_dist.item())
                        npos_dist += 1

                        det_gt[ious.argmax()] = True
                    else:
                        FP.append(1)
                        TP.append(0)
                else:
                    FP.append(1)
                    TP.append(0)

        TP = np.array(TP)
        FP = np.array(FP)
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        [ap, mpre, mrec, ii] = self.calculateAveragePrecision(rec, prec)

        ret_metrics = {
            'ap': ap,
            'prec': prec,
            'rec': rec,
            'npos': npos,
            'npos_dist': npos_dist,
            'TP': TP,
            'FP': FP,
            'acc_TP': acc_TP,
            'acc_FP': acc_FP,
            'dist': np.mean(dists)
        }

        return ret_metrics