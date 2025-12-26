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

from torch.nn import functional as F

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

class GroupCost():
    def __init__(self):
        pass

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

    def cal_group_IoU(self, gt: torch.Tensor, pred: torch.Tensor):
        gt_grp_id = torch.where(gt==1)[0]
        gt_non_grp_id = torch.where(gt==0)[0]
        pred_grp_id = torch.where(pred==1)[0]
        pred_non_grp_id = torch.where(pred==0)[0]

        intersection = len(set(gt_grp_id.cpu().numpy()).intersection(set(pred_grp_id.cpu().numpy())))
        union = len(set(gt_grp_id.cpu().numpy()).union(set(pred_grp_id.cpu().numpy())))
        if union==0:
            iou = torch.tensor(0.0, device=gt.device)
        else:
            iou = intersection / union

        return iou
    
    def cal_coatt_dist(self, hm_pred: torch.Tensor, hm_gt: torch.Tensor):
        hm_pred_pnt = spatial_argmax2d(hm_pred, normalize=True)
        hm_gt_pnt = spatial_argmax2d(hm_gt, normalize=True)
        hm_dist = (hm_gt_pnt - hm_pred_pnt).pow(2).sum(-1).sqrt()

        return hm_dist

    def compute(self, grouping_pred: torch.Tensor, grouping_gt: torch.Tensor, hm_pred: torch.Tensor, hm_gt: torch.Tensor):
        group_ious = []
        dists = []

        batch_num, coatt_num_gt, people_num = grouping_gt.shape
        _, coatt_num_pred, _ = grouping_pred.shape

        grouping_empty = torch.zeros((1, people_num), device=grouping_gt.device)
        coatt_empty = torch.zeros((1, 2), device=grouping_gt.device)

        hm_h, hm_w = hm_pred.shape[2], hm_pred.shape[3]
        coatt_pt_gt = spatial_argmax2d(hm_gt.reshape(batch_num*coatt_num_gt, hm_h, hm_w), normalize=True).view(batch_num, coatt_num_gt, -1)
        coatt_pt_pred = spatial_argmax2d(hm_pred.reshape(batch_num*coatt_num_pred, hm_h, hm_w), normalize=True).view(batch_num, coatt_num_pred, -1)

        for b in range(batch_num):
            grouping_gt_b = grouping_gt[b]
            grouping_pred_b = grouping_pred[b]
            coatt_pt_gt_b = coatt_pt_gt[b]
            coatt_pt_pred_b = coatt_pt_pred[b]

            # remove empty groups
            gt_b_mask = grouping_gt_b.sum(dim=-1)>0
            grouping_gt_b = grouping_gt_b[gt_b_mask]
            grouping_gt_b = torch.cat([grouping_gt_b, grouping_empty], dim=0)
            coatt_pt_gt_b = coatt_pt_gt_b[gt_b_mask]
            coatt_pt_gt_b = torch.cat([coatt_pt_gt_b, coatt_empty], dim=0)

            # remove non-group predictions
            pred_b_mask = grouping_pred_b.sum(dim=-1)>0
            grouping_pred_b = grouping_pred_b[pred_b_mask]
            coatt_pt_pred_b = coatt_pt_pred_b[pred_b_mask]

            # add empty group if no predictions
            if grouping_pred_b.shape[0]==0:
                grouping_pred_b = grouping_empty
                coatt_pt_pred_b = coatt_empty

            # get numbers of groups
            pred_num = grouping_pred_b.shape[0]
            gt_num = grouping_gt_b.shape[0]

            # create IoU matrix
            group_iou_matrix = torch.zeros((1, gt_num, pred_num), device=grouping_gt.device)
            dist_matrix = torch.zeros((1, gt_num, pred_num), device=grouping_gt.device)
            for gt_idx in range(gt_num):
                for pred_idx in range(pred_num):
                    # compute IoU
                    gt = grouping_gt_b[gt_idx]
                    pred = grouping_pred_b[pred_idx]
                    iou = self.cal_group_IoU(gt, pred)
                    group_iou_matrix[0, gt_idx, pred_idx] = iou

                    # compute co-attention distance
                    coatt_pt_gt_one = coatt_pt_gt_b[gt_idx]
                    coatt_pt_pred_one = coatt_pt_pred_b[pred_idx]
                    dist = torch.norm(coatt_pt_gt_one - coatt_pt_pred_one, p=2)
                    dist_matrix[0, gt_idx, pred_idx] = dist

            # solve the linear assignment problem
            cost_matrix = -group_iou_matrix
            assignment = batch_linear_assignment(cost_matrix)
            row_ind, col_ind = assignment_to_indices(assignment)
            
            # compute the cost for the assigned pairs
            cost_minimums = cost_matrix[torch.arange(cost_matrix.shape[0]).unsqueeze(1), row_ind, col_ind]
            group_iou_mean = torch.mean(cost_minimums) * -1.0  # mean IoU over assigned pairs
            group_ious.append(group_iou_mean.item())

            # compute the co-attention distance for the assigned pairs
            dist_assigned = dist_matrix[torch.arange(dist_matrix.shape[0]).unsqueeze(1), row_ind, col_ind]
            dist_mean = torch.mean(dist_assigned)
            dists.append(dist_mean.item())

        group_ious_mean = np.mean(group_ious)
        dists_mean = np.mean(dists)

        ret_metrics = {
            'Group_IoU': group_ious_mean,
            'Group_Dist': dists_mean
        }

        return ret_metrics
    

class GroupAP():
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

    def cal_group_IoU(self, gt: torch.Tensor, pred: torch.Tensor):
        gt_grp_id = torch.where(gt==1)[0]
        gt_non_grp_id = torch.where(gt==0)[0]
        pred_grp_id = torch.where(pred==1)[0]
        pred_non_grp_id = torch.where(pred==0)[0]

        intersection = len(set(gt_grp_id.cpu().numpy()).intersection(set(pred_grp_id.cpu().numpy())))
        union = len(set(gt_grp_id.cpu().numpy()).union(set(pred_grp_id.cpu().numpy())))
        if union==0:
            iou = torch.tensor(0.0, device=gt.device)
        else:
            iou = intersection / union
        
        return iou

    
    def cal_coatt_dist(self, hm_pred: torch.Tensor, hm_gt: torch.Tensor):
        hm_pred_pnt = spatial_argmax2d(hm_pred, normalize=True)
        hm_gt_pnt = spatial_argmax2d(hm_gt, normalize=True)
        hm_dist = (hm_gt_pnt - hm_pred_pnt).pow(2).sum(-1).sqrt()

        return hm_dist

    def compute(self, grouping_pred: torch.Tensor, grouping_gt: torch.Tensor, hm_pred: torch.Tensor, hm_gt: torch.Tensor):
        npos = 0
        TP = []
        FP = []
        npos_dist = 0
        dists = []

        batch_all = len(grouping_gt)
        confidence_scores_all = torch.tensor([], device=grouping_gt[0].device)
        for b in range(batch_all):
            # get the ground-truth groups
            grouping_gt_b = grouping_gt[b]
            _, g_token_num_ori, people_num = grouping_gt_b.shape
            grouping_gt_b = grouping_gt_b.view(g_token_num_ori, people_num)
            grouping_gt_b = grouping_gt_b[grouping_gt_b.sum(dim=-1)>1]
            npos += grouping_gt_b.shape[0]

            # get the predicted groups
            grouping_pred_b = grouping_pred[b]
            grouping_pred_b = grouping_pred_b.view(-1, people_num)

            # get the heatmaps
            hm_pred_b = hm_pred[b]
            hm_gt_b = hm_gt[b]
            _, _, H, W = hm_pred_b.shape
            hm_pred_b = hm_pred_b.view(-1, H, W)
            hm_gt_b = hm_gt_b.view(-1, H, W)

            group_flag = grouping_pred_b.sum(dim=-1) > 1
            if group_flag.sum() == 0:
                continue
            grouping_pred_b = grouping_pred_b[group_flag]
            hm_pred_b = hm_pred_b[group_flag]

            # compute confidence scores for each predicted group
            # grouping_pred_b_conf = torch.mean(grouping_pred_b, dim=-1, dtype=torch.float32)
            # confidence_scores = torch.cat([confidence_scores, grouping_pred_b_conf])

            # compute confidence scores for each predicted group
            hm_pred_b_flat = hm_pred_b.view(hm_pred_b.shape[0], H*W)
            hm_pred_b_peak_vals, _ = torch.max(hm_pred_b_flat, dim=-1)
            confidence_scores = hm_pred_b_peak_vals
            confidence_scores_all = torch.cat([confidence_scores_all, confidence_scores])

            # sort predictions by confidence scores
            confidence_scores, sorted_indices = torch.sort(confidence_scores, descending=True)
            grouping_pred_b = grouping_pred_b[sorted_indices]
            hm_pred_b = hm_pred_b[sorted_indices]

            # match the predictions to ground-truth
            det_gt = torch.zeros(grouping_gt_b.shape[0], dtype=torch.bool, device=grouping_gt_b.device)
            for pred_idx in range(grouping_pred_b.shape[0]):
                pred = grouping_pred_b[pred_idx]
                ious = []
                for gt_idx in range(grouping_gt_b.shape[0]):
                    gt = grouping_gt_b[gt_idx]
                    iou = self.cal_group_IoU(gt, pred)
                    ious.append(iou)
                ious = torch.tensor(ious)
                max_iou = ious.max() if len(ious)>0 else torch.tensor(0.0)

                if max_iou >= self.iou_thresh:
                    if not det_gt[ious.argmax()]:
                        TP.append(1)
                        FP.append(0)

                        # mark gt as detected
                        det_gt[ious.argmax()] = True

                        # compute hm distance if matched
                        hm_pred_target = hm_pred_b[pred_idx]
                        hm_gt_target = hm_gt_b[ious.argmax()]
                        hm_dist = self.cal_coatt_dist(hm_pred_target.unsqueeze(0), hm_gt_target.unsqueeze(0))
                        dists.append(hm_dist.item())
                        npos_dist += 1
                    else:
                        FP.append(1)
                        TP.append(0)
                else:
                    FP.append(1)
                    TP.append(0)

        # sort TP and FP based on confidence scores
        confidence_scores_all, sorted_indices = torch.sort(confidence_scores_all, descending=True)
        TP = torch.tensor(TP, device=confidence_scores_all.device)[sorted_indices].cpu().numpy()
        FP = torch.tensor(FP, device=confidence_scores_all.device)[sorted_indices].cpu().numpy()

        # Compute cumulative sums
        acc_TP = np.cumsum(TP)
        acc_FP = np.cumsum(FP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        ap, mpre, mrec, ii = self.calculateAveragePrecision(rec.tolist(), prec.tolist())

        ret_metrics = {
            'ap': ap,
            'prec': prec,
            'rec': rec,
            'npos': npos,
            'npos_dist': npos_dist,
            'dist': np.mean(dists)
        }

        return ret_metrics
