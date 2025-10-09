from typing import List, Tuple
import numpy as np
import sys

from torch_linear_assignment import batch_linear_assignment, assignment_to_indices
import kornia.geometry.subpix as kgs
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rinnegan_loss(gaze_heatmap_gt, gaze_vec_gt, inout_gt, gaze_heatmap_pred, gaze_vec_pred, inout_pred, epoch=None):
    heatmap_loss = torch.tensor(0.0)
    angular_loss = torch.tensor(0.0)
    
    if torch.sum(inout_gt) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
        heatmap_loss = compute_heatmap_loss(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt)
        angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, inout_gt)
        
    bce_loss = compute_bce_loss(inout_pred, inout_gt)
    total_loss = 1000 * heatmap_loss + 3 * angular_loss # + 2 * bce_loss

    logs = {
        "heatmap_loss": heatmap_loss.item(),
        "angular_loss": angular_loss.item(),
        "bce_loss": bce_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs

def compute_chong_loss(gaze_heatmap_gt, inout_gt, gaze_heatmap_pred, inout_pred, epoch=None):
    heatmap_loss = torch.tensor(0.0)

    if torch.sum(inout_gt) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
        heatmap_loss = compute_heatmap_loss(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt)
        
    bce_loss = compute_bce_loss(inout_pred, inout_gt)
    total_loss = 1000 * heatmap_loss # + 2 * bce_loss

    logs = {
        "heatmap_loss": heatmap_loss.item(),
        "bce_loss": bce_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs


def social_loss(social_pred, social_gt, mask, pos_weight=2.0):
    """ Compute a loss for coatt or laeo or lah. This implements a standard binary cross-entropy loss.
    
    Args:
        social_pred: tensor representing the predicted social gaze logits.
        social_gt: tensor representing the ground-truth social gaze binary labels.
        mask: a binary tensor denoting the positions of valid predictions to keep in the loss. This is 
        used to discard social gaze pairs where one side is a padded person (ie. black head image).
        
    Returns:
        Tensor representing the loss value (including the corresponding computation graph)
        Dictionary representing the items to log (e.g. {"total_loss": total_loss})
    """

    num_instances = mask.sum()
    loss = F.binary_cross_entropy_with_logits(social_pred, social_gt, pos_weight=torch.tensor(pos_weight, device=social_gt.device), reduction="none")
    loss = torch.mul(loss, mask).sum() / (num_instances + 1e-6)

    return loss


def compute_social_loss(lah_pred, lah_gt, lah_mask, laeo_pred, laeo_gt, laeo_mask, coatt_pred, coatt_gt, coatt_mask):
    
    lah_loss = social_loss(lah_pred, lah_gt, lah_mask, pos_weight=3.0)
    laeo_loss = social_loss(laeo_pred, laeo_gt, laeo_mask)
    coatt_loss = social_loss(coatt_pred, coatt_gt, coatt_mask)
    
    lah_coeff = 1
    laeo_coeff = 1
    coatt_coeff = 1
#     total_loss = laeo_coeff*laeo_loss
    total_loss = lah_coeff*lah_loss + laeo_coeff*laeo_loss + coatt_coeff*coatt_loss 

    logs = {
        "laeo_loss": laeo_loss.item(),
        "lah_loss": lah_loss.item(),
        "coatt_loss": coatt_loss.item(),
        "total_loss": total_loss.item(), 
    }

    return total_loss, logs

def compute_interact_loss(gaze_vec_gt, gaze_hm_gt, inout_gt, gaze_vec_pred, gaze_hm_pred, inout_pred, dataset=None, epoch=None):
    heatmap_loss = torch.tensor(0.0).to(gaze_hm_pred.device)
    angular_loss = torch.tensor(0.0).to(gaze_hm_pred.device)
    dist_loss = torch.tensor(0.0).to(gaze_hm_pred.device)
    inout_loss = torch.tensor(0.0).to(gaze_hm_pred.device)
    
    mask = (inout_gt==1)
    if torch.sum(mask) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
        angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, mask, dataset)
        heatmap_loss = compute_heatmap_loss(gaze_hm_pred, gaze_hm_gt, mask, dataset)
#         dist_loss = compute_dist_loss(gaze_pt_pred, gaze_pt_gt, mask)
    
    mask = (inout_gt!=-1)
    if torch.sum(mask) > 0:
        inout_loss = compute_inout_loss(inout_pred, inout_gt, mask)
#     total_loss = 20 * angular_loss + 1000 * heatmap_loss    # for GeomGaze
    total_loss = 3 * angular_loss + 100 * dist_loss + 1000 * heatmap_loss + 2 * inout_loss    # for GazeInteract

    logs = {
        "heatmap_loss": heatmap_loss.item(),
        "dist_loss": dist_loss.item(),
        "inout_loss": inout_loss.item(),
        "angular_loss": angular_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs

def compute_coatt_loss(coatt_hm_gt, coatt_hm_pred, coatt_level_gt, coatt_level_pred, person_tokens):
    if len(coatt_hm_gt.shape) == 4:  # if the input is a single frame
        b, coatt_num, h, w = coatt_hm_gt.shape
        t = 1
    elif len(coatt_hm_gt.shape) == 5:  # if the input is a sequence of frames
        b, t, coatt_num, h, w = coatt_hm_gt.shape

    # compute cost matrix for coatt heatmaps
    coatt_hm_gt = coatt_hm_gt.view(b*t, coatt_num, h*w)
    coatt_hm_pred = coatt_hm_pred.view(b*t, coatt_num, h*w)
    cost_matrix_hm = torch.cdist(coatt_hm_gt, coatt_hm_pred, p=2)
    # print('cost_matrix_hm', cost_matrix_hm.shape)

    # compute cost matrix for coatt binary classification (bce loss, b*t, coatt_num, coatt_num)
    coatt_level_gt = coatt_level_gt.view(b*t, coatt_num, -1)  # (b*t, coatt_num, people_num)
    coatt_level_pred = coatt_level_pred.view(b*t, coatt_num, -1)  # (b*t, coatt_num, people_num)
    coatt_level_gt_expand = coatt_level_gt.unsqueeze(2).expand(-1, -1, coatt_level_gt.shape[1], -1).float()  # (b*t, coatt_num, coatt_num, people_num)
    coatt_level_pred_expand = coatt_level_pred.unsqueeze(1).expand(-1, coatt_level_pred.shape[1], -1, -1)  # (b*t, coatt_num, coatt_num, people_num)
    bce_loss = F.binary_cross_entropy_with_logits(coatt_level_pred_expand, coatt_level_gt_expand, reduction="none")
    cost_matrix_level = bce_loss.mean(dim=-1)  # average over people_num dimension
    # print('cost_matrix_level', cost_matrix_level.shape)

    # combine all cost matrices
    cost_matrix = cost_matrix_hm + cost_matrix_level

    # solve the linear assignment problem
    assignment = batch_linear_assignment(cost_matrix)
    row_ind, col_ind = assignment_to_indices(assignment)
    
    # compute the cost for the assigned pairs
    cost_minimums = cost_matrix[torch.arange(cost_matrix.shape[0]).unsqueeze(1), row_ind, col_ind]
    coatt_loss = torch.mean(cost_minimums)
    cost_minimums_hm = cost_matrix_hm[torch.arange(cost_matrix_hm.shape[0]).unsqueeze(1), row_ind, col_ind]
    cost_loss_hm = torch.mean(cost_minimums_hm)
    cost_minimums_level = cost_matrix_level[torch.arange(cost_matrix_level.shape[0]).unsqueeze(1), row_ind, col_ind]
    cost_loss_level = torch.mean(cost_minimums_level)

    # compute a loss function in which person_tokes joining the same group are close in the feature space
    coatt_level_gt = coatt_level_gt.view(b, t, coatt_num, -1)  # (b, t, coatt_num, people_num)

    temp_param = 0.2
    con_loss = 0.0
    for b_idx in range(b):
        for t_idx in range(t):
            for coatt_idx in range(coatt_num):
                coatt_level_gt_curr = coatt_level_gt[b_idx, t_idx, coatt_idx]  # (people_num,)
                if torch.sum(coatt_level_gt_curr) <= 1:
                    continue
                person_tokens_all = person_tokens[b_idx, t_idx]  # (n, token_dim)
                person_tokens_grp = person_tokens_all[coatt_level_gt_curr==1]  # (num_in_group, token_dim)
                for person_idx in range(coatt_level_gt_curr.shape[0]):
                    person_tokens_target = person_tokens[b_idx, t_idx, person_idx]  # (token_dim,)
                    con_loss_den = torch.sum(torch.exp(torch.cosine_similarity(person_tokens_target.unsqueeze(0), person_tokens_all, dim=-1)) / temp_param).sum()
                    con_loss_mol = torch.sum(torch.exp(torch.cosine_similarity(person_tokens_target.unsqueeze(0), person_tokens_grp, dim=-1)) / temp_param).sum()
                    con_loss_curr = - torch.log((con_loss_mol + 1e-6) / (con_loss_den + 1e-6))
                    con_loss += con_loss_curr
    con_loss = con_loss / (b * t * coatt_num * coatt_level_gt.shape[-1])

    logs = {
        "coatt_loss": coatt_loss.item(),
        "coatt_hm_loss": cost_loss_hm.item(),
        "coatt_level_loss": cost_loss_level.item(),
        "con_loss": con_loss,
    }

    return coatt_loss, con_loss, logs

# def compute_coatt_loss(coatt_hm_gt, coatt_hm_pred, coatt_level_gt, coatt_level_pred):
#     # Check shape to handle both single frame and sequence of frames
#     if len(coatt_hm_gt.shape) == 5:
#         b, t, coatt_num, h, w = coatt_hm_gt.shape
#         coatt_hm_gt = coatt_hm_gt.view(b * t, coatt_num, h, w)
#         coatt_hm_pred = coatt_hm_pred.view(b * t, coatt_num, h, w)
#         coatt_level_gt = coatt_level_gt.view(b * t, coatt_num, -1)
#         coatt_level_pred = coatt_level_pred.view(b * t, coatt_num, -1)
#     else: # len(coatt_hm_gt.shape) == 4
#         b, coatt_num, h, w = coatt_hm_gt.shape
#         t = 1
#         coatt_level_gt = coatt_level_gt.view(b, coatt_num, -1)
#         coatt_level_pred = coatt_level_pred.view(b, coatt_num, -1)
    
#     # Flatten the heatmaps for cdist
#     coatt_hm_gt_flat = coatt_hm_gt.view(b * t, coatt_num, h * w)
#     coatt_hm_pred_flat = coatt_hm_pred.view(b * t, coatt_num, h * w)

#     # Compute cost matrix for coatt heatmaps
#     cost_matrix_hm = torch.cdist(coatt_hm_gt_flat, coatt_hm_pred_flat, p=2)

#     # Compute cost matrix for coatt levels using efficient broadcasting
#     cost_matrix_level = F.binary_cross_entropy_with_logits(
#         coatt_level_pred.unsqueeze(1).expand(-1, coatt_num, -1, -1),
#         coatt_level_gt.unsqueeze(2).expand(-1, -1, coatt_num, -1).float(),
#         reduction="none"
#     ).mean(dim=-1)

#     # Combine all cost matrices
#     cost_matrix = cost_matrix_hm + cost_matrix_level

#     # Solve the linear assignment problem
#     assignment = batch_linear_assignment(cost_matrix)
#     row_ind, col_ind = assignment_to_indices(assignment)
    
#     # Use torch.gather for efficient indexing
#     cost_minimums = torch.gather(cost_matrix, 1, col_ind.unsqueeze(1).repeat(1, 1, coatt_num)).squeeze(1)
#     coatt_loss = cost_minimums.mean()

#     # Compute individual loss components more efficiently
#     cost_loss_hm = torch.gather(cost_matrix_hm, 1, col_ind.unsqueeze(1).repeat(1, 1, coatt_num)).squeeze(1).mean()
#     cost_loss_level = torch.gather(cost_matrix_level, 1, col_ind.unsqueeze(1).repeat(1, 1, coatt_num)).squeeze(1).mean()

#     logs = {
#         "coatt_loss": coatt_loss.item(),
#         "coatt_hm_loss": cost_loss_hm.item(),
#         "coatt_level_loss": cost_loss_level.item(),
#     }

#     return coatt_loss, logs

def compute_sharingan_loss(gaze_vec_gt, gaze_pt_gt, inout_gt, gaze_vec_pred, gaze_pt_pred, inout_pred, epoch=None):
    heatmap_loss = torch.tensor(0.0)
    angular_loss = torch.tensor(0.0)
    dist_loss = torch.tensor(0.0)
    inout_loss = torch.tensor(0.0)
    
    mask = (inout_gt==1)
    if torch.sum(mask) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
        angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, mask)
        dist_loss = compute_dist_loss(gaze_pt_pred, gaze_pt_gt, mask)
        
    mask = (inout_gt!=-1)
    if torch.sum(mask) > 0:
        inout_loss = compute_inout_loss(inout_pred, inout_gt, mask)
    total_loss = 3 * angular_loss + 100 * dist_loss + 1000 * heatmap_loss+ 2 * inout_loss

    logs = {
        "heatmap_loss": heatmap_loss.item(),
        "dist_loss": dist_loss.item(),
        "inout_loss": inout_loss.item(),
        "angular_loss": angular_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs


def compute_gf_loss(gaze_vec_gt, gaze_vec_pred, inout_gt, inout_pred=None, gaze_pt_gt=None, gaze_pt_pred=None, gaze_heatmap_gt=None, gaze_heatmap_pred=None, epoch=None):
    heatmap_loss = torch.tensor(0.0)
    angular_loss = torch.tensor(0.0)
    dist_loss = torch.tensor(0.0)

    if torch.sum(inout_gt) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
        angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, inout_gt)
        if gaze_pt_pred is not None:
            dist_loss = compute_dist_loss(gaze_pt_pred, gaze_pt_gt, inout_gt)

        if gaze_heatmap_pred is not None:
            heatmap_loss = compute_heatmap_loss(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt)

    total_loss = 3 * angular_loss + 100 * dist_loss + 1000 * heatmap_loss  # 100 or 3 * angular_loss

    logs = {
        "angular_loss": angular_loss.item(),
        "heatmap_loss": heatmap_loss.item(),
        "dist_loss": dist_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs


def compute_vat_loss(gaze_vec_gt, gaze_vec_pred, inout_gt, inout_pred=None, gaze_pt_gt=None, gaze_pt_pred=None, gaze_heatmap_gt=None, gaze_heatmap_pred=None, epoch=None):
    heatmap_loss = torch.tensor(0.0)
    angular_loss = torch.tensor(0.0)
    dist_loss = torch.tensor(0.0)

    if torch.sum(inout_gt) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
        angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, inout_gt)
        if gaze_pt_pred is not None:
            dist_loss = compute_dist_loss(gaze_pt_pred, gaze_pt_gt, inout_gt)
        if gaze_heatmap_pred is not None:
            heatmap_loss = compute_heatmap_loss(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt)

    bce_loss = compute_bce_loss(inout_pred, inout_gt)

    total_loss = 2 * bce_loss
    # total_loss = 3 * angular_loss + 100 * dist_loss + 1000 * heatmap_loss + 2 * bce_loss # 2 * bce_loss
    # total_loss = 3 * angular_loss + 100 * dist_loss + 1000 * heatmap_loss + 0.1 * bce_loss # 2 * bce_loss
    # total_loss = 7 * (50 * dist_loss) + 1 * (10 * angular_loss) + 2 * (1 * bce_loss) + 1000 * heatmap_loss

    logs = {
        "heatmap_loss": heatmap_loss.item(),
        "angular_loss": angular_loss.item(),
        "dist_loss": dist_loss.item(),
        "bce_loss": bce_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs


def compute_dist_loss(gp_pred, gp_gt, mask):
    dist_loss = (gp_pred - gp_gt).pow(2).sum(dim=-1)
    dist_loss = torch.mul(dist_loss, mask)
    dist_loss = torch.sum(dist_loss) / torch.sum(mask)
    return dist_loss


def compute_heatmap_loss(hm_pred, hm_gt, mask, dataset=None):
    heatmap_loss = F.mse_loss(hm_pred, hm_gt, reduce=False).mean([2, 3])
    heatmap_loss = torch.mul(heatmap_loss, mask)
    if dataset:
        dataset_mask = np.where((np.array(dataset)=='coatt').astype(np.int) + (np.array(dataset)=='laeo').astype(np.int))[0]
        fact = torch.zeros_like(heatmap_loss) + 1
        fact[dataset_mask] = 0.1    # 0.1x loss for UCO-LAEO and VideoCoAtt
        heatmap_loss = heatmap_loss * fact
    heatmap_loss = torch.sum(heatmap_loss) / torch.sum(mask)
    return heatmap_loss


def compute_angular_loss(gv_pred, gv_gt, mask, dataset=None):
    angular_loss = (1 - (gv_pred*gv_gt).sum(axis=-1)) / 2
    angular_loss = torch.mul(angular_loss, mask)
    if dataset:
        dataset_mask = np.where((np.array(dataset)=='coatt').astype(np.int) + (np.array(dataset)=='laeo').astype(np.int))[0]
        fact = torch.zeros_like(angular_loss) + 1
        fact[dataset_mask] = 0.1    # 0.1x loss for UCO-LAEO and VideoCoAtt
        angular_loss = angular_loss * fact
    angular_loss = torch.sum(angular_loss) / torch.sum(mask)
    return angular_loss


def compute_inout_loss(io_pred, io_gt, mask):
    io_gt = (io_gt * mask)
    bce_loss = F.binary_cross_entropy_with_logits(io_pred, io_gt, reduction="none")
    bce_loss = (bce_loss * mask).sum() / mask.sum()
    
    return bce_loss


def infer_gp_from_hm(hm_pred):
    # Compute soft argmax of heatmaps
    gp_pred = kgs.spatial_soft_argmax2d(hm_pred.unsqueeze(1), temperature=torch.tensor(10000.0), normalized_coordinates=True)
    gp_pred = (1 + gp_pred.squeeze(1)) / 2  # [-1, 1] >> [0, 1]
    return gp_pred
