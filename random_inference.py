from src.metrics import GroupIoU
import torch

batch_size = 1
token_num = 1
people_num = 4

coatt_levels_gt = torch.zeros((batch_size, token_num, people_num), dtype=torch.long)
coatt_levels_gt[0, 0, 0] = 1
coatt_levels_gt[0, 0, 1] = 1
coatt_levels_gt[0, 0, 2] = 1
print(f'coatt_levels_gt: {coatt_levels_gt}')

ious = []
for _ in range(10):
    coatt_level_pred_label = torch.randint(0, 2, (batch_size, token_num, people_num))
    print(f'coatt_level_pred_label: {coatt_level_pred_label}')
    
    group_iou_func = GroupIoU()
    group_iou_func.gen_matrix(coatt_level_pred_label, coatt_levels_gt)
    group_iou_func.match()
    group_iou_pairs, group_iou_pairs_masked, group_pred_pairs_opt_masked, grouping_gt_masked = group_iou_func.compute()
    
    group_iou_pairs_mean = group_iou_pairs.mean().item()
    print(f'Group IoU: {group_iou_pairs_mean:.4f}')
    group_iou_pairs_masked_mean = group_iou_pairs_masked.mean().item()
    print(f'Group IoU (masked): {group_iou_pairs_masked_mean:.4f}')
    ious.append((group_iou_pairs_mean, group_iou_pairs_masked_mean))

print(f'Average Group IoU over 10 runs: {sum([iou[0] for iou in ious]) / len(ious):.4f}')
print(f'Average Group IoU (masked) over 10 runs: {sum([iou[1] for iou in ious]) / len(ious):.4f}')