import os
import glob
import json
import pandas as pd
import pickle
import mmap
import numpy as np
import cv2
import torch

def fast_pickle_load(file_path):
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return pickle.loads(mm)

# set result directory
result_dir = os.path.join('checkpoints')

# set models
model_names = []
model_names_omit = []

# set dataset type
# dataset_type = 'videocoatt'
# dataset_type = 'videoattentiontarget'
dataset_type = 'childplay'

if dataset_type == 'videocoatt':
    dataset_dir = os.path.join('data', 'VideoCoAtt_Dataset', 'images_nk')
    coatt_level_thresh = 0.001
elif dataset_type == 'videoattentiontarget':
    dataset_dir = os.path.join('data', 'videoattentiontarget', 'images')
    coatt_level_thresh = 0.001
elif dataset_type == 'childplay':
    dataset_dir = os.path.join('data', 'ChildPlay-gaze', 'images')
    coatt_level_thresh = 0.4

model_name = "17-23-01_combined_social_COA_True_True_bce_SOC_True_hm_coef_iter_temp_2-3_1_1e-05_w_gaze_vec_FRZ_IT_VE"
model_names.append(model_name)
model_names_omit.append('Ours')

# create save directory
save_dir = os.path.join('analysis', 'visualization', dataset_type, f'{os.path.basename(__file__)[:-3]}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_data_ious = []
save_data_dists = []

results_all = {}
for model_name, model_name_omit in zip(model_names, model_names_omit):
    print(f"Results for {model_name}")

    # load results
    result_pickle_paths = glob.glob(os.path.join(result_dir, '*', '*', model_name, dataset_type, 'test-predictions.pickle'))
    result_pickle_path = result_pickle_paths[0]
    results = fast_pickle_load(result_pickle_path)
    results_all[model_name_omit] = results

save_names = []
results = results_all[model_names_omit[0]]
for idx, result in enumerate(results):
    # retrieve sample path
    paths = result['path']
    sample_path = paths[len(paths)//2][0]  # middle frame path
    save_names.append(sample_path)

    # retrieve group IoUs
    group_iou_pairwise = result['group_iou_pairwise'].item()
    group_iou_pp = result['group_iou_pp'].item()
    group_iou_grp = result['group_iou_grp'].item()

    # retrieve group distances
    group_dist_pairwise = result['group_dist_pairwise'].item()
    group_dist_pp = result['group_dist_pp'].item()
    group_dist_grp = result['group_dist_grp'].item()

    save_data_ious_result = []
    save_data_dists_result = []
    for model_name_omit in model_names_omit:
        save_data_ious_result.append(group_iou_grp)
        save_data_dists_result.append(group_dist_grp)
    
    save_data_ious.append(save_data_ious_result)
    save_data_dists.append(save_data_dists_result)

print(f'Load {len(results)} results for each model.')

# generate dataframes
df_ious = pd.DataFrame(np.array(save_data_ious).reshape(-1, len(model_names)), columns=model_names_omit, index=save_names)
df_dists = pd.DataFrame(np.array(save_data_dists).reshape(-1, len(model_names)), columns=model_names_omit, index=save_names)
print("IoU DataFrame:")
print(df_ious)

# find data in which Ours outperforms MTGS-PP by more than 0.2 IoU
iou_diff_threshold = 0.2
dist_diff_threshold = 20.0

# initialize all indices as selected
selected_indices = df_ious.index
selected_indices_soc = df_ious.index[(df_ious['Ours'] < iou_diff_threshold)]

# randomly select 10 samples from the selected indices
np.random.seed(777)
if len(selected_indices) > 10:
    selected_indices = np.random.choice(selected_indices, 10, replace=False)

df_selected_ious = df_ious.loc[selected_indices]
df_selected_dists = df_dists.loc[selected_indices]
# print("Selected IoU DataFrame:")
# print(df_selected_ious)
# print("Selected Distance DataFrame:")
# print(df_selected_dists)

group_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128)]

for data_id in selected_indices:
    if dataset_type == 'videoattentiontarget':
        idx_top = data_id.split('/')[0].replace('_', ' ')
        idx = os.path.join(idx_top, *data_id.split('/')[1:])
    elif dataset_type == 'videocoatt':
        idx = data_id
    elif dataset_type == 'childplay':
        idx = '/'.join(data_id.split('/')[1:])

    save_dir_data = os.path.join(save_dir, data_id.replace('/', '_'))
    if not os.path.exists(save_dir_data):
        os.makedirs(save_dir_data)
    
    print(f"Visualizing {data_id}...")

    # visualize results for each model
    for model_name_omit in model_names_omit:
        result = results_all[model_name_omit]
        result_sample = next((res for res in result if res['path'][len(result[0]['path'])//2][0] == data_id), None)

        # read images 
        img_path = os.path.join(dataset_dir, idx)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        img_vis = img.copy()
        img_vis_only_bboxes = img.copy()

        # save the original image
        save_path = os.path.join(save_dir_data, f"img.jpg")
        cv2.imwrite(save_path, img_vis) 

        # retrieve predicted co-attention points
        coatt_level_pred = torch.sigmoid(result_sample['coatt_level_pred'][0])
        coatt_level_flag = torch.sum(coatt_level_pred > coatt_level_thresh, dim=-1) > 1
        coatt_hm_pred_pk_cd = result_sample['coatt_hm_pred_pk_cd'][0][coatt_level_flag]
        coatt_level_pred = coatt_level_pred[coatt_level_flag]

        # visualize predicted co-attention points on the image
        head_boxes = result_sample['head_bboxes'][0]

        for grp_idx, pk_cd in enumerate(coatt_hm_pred_pk_cd):
            pk_x = int(pk_cd[0].item() * img_w)
            pk_y = int(pk_cd[1].item() * img_h)
            grp_color = group_colors[grp_idx % len(group_colors)]
            grp_member = 0

            # plot head boxes with high co-attention level for the group
            coatt_level_pred_grp = coatt_level_pred[grp_idx]

            for head_idx in range(coatt_level_pred_grp.shape[0]):
                if coatt_level_pred_grp[head_idx] > coatt_level_thresh:
                    head_box = head_boxes[head_idx]
                    if torch.sum(head_box) == 0:
                        continue  # skip if head box is not valid
                    grp_member += 1

            if grp_member > 1:

                # plot the predicted co-attention point
                cv2.circle(img_vis, (pk_x, pk_y), 10, grp_color, -1)  # colored circle for predicted co-attention points
                cv2.circle(img_vis_only_bboxes, (pk_x, pk_y), 10, grp_color, -1)  # colored circle for predicted co-attention points on the image with only bounding boxes

                for head_idx in range(coatt_level_pred_grp.shape[0]):
                    if coatt_level_pred_grp[head_idx] > coatt_level_thresh:
                        head_box = head_boxes[head_idx]
                        if torch.sum(head_box) == 0:
                            continue  # skip if head box is not valid
                        x1 = int(head_box[0].item() * img_w)
                        y1 = int(head_box[1].item() * img_h)
                        x2 = int(head_box[2].item() * img_w)
                        y2 = int(head_box[3].item() * img_h)

                        # change color based on group index
                        cv2.rectangle(img_vis, (x1, y1), (x2, y2), grp_color, 2)  # colored box for heads with high co-attention level
                        cv2.rectangle(img_vis_only_bboxes, (x1, y1), (x2, y2), grp_color, 2)  # colored box for heads with high co-attention level on the image with only bounding boxes

                        # plot coattention level as text above the head box
                        text = f"{coatt_level_pred_grp[head_idx].item():.2f}"
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2
                        text_y = y1 - 10
                        cv2.putText(img_vis, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # colored text for co-attention level

        # plot gaze target point if gaze vector is available
        gaze_pts_pred = result_sample['gaze_pts_pred'][0]
        gv_pred = result_sample['gv_pred'][0]
        for head_ibx in range(head_boxes.shape[0]):
            head_box = head_boxes[head_ibx]
            if torch.sum(head_box) == 0:
                continue  # skip if head box is not valid
            x1 = int(head_box[0].item() * img_w)
            y1 = int(head_box[1].item() * img_h)
            x2 = int(head_box[2].item() * img_w)
            y2 = int(head_box[3].item() * img_h)

            gaze_pts_pred_head = gaze_pts_pred[head_ibx]
            if torch.sum(gaze_pts_pred_head) == 0:
                continue  # skip if gaze point is not valid
            gaze_x = int(gaze_pts_pred_head[0].item() * img_w)
            gaze_y = int(gaze_pts_pred_head[1].item() * img_h)

            # plot the gaze target arrow
            start = ((x1 + x2) // 2, (y1 + y2) // 2)
            end = (gaze_x, gaze_y)
            # cv2.arrowedLine(img_vis, start, end, (255, 255, 255), 2, cv2.LINE_AA, 0, 0.1)

        # save the visualization
        save_path = os.path.join(save_dir_data, f"{model_name_omit}.jpg")
        cv2.imwrite(save_path, img_vis)
        save_path = os.path.join(save_dir_data, f"{model_name_omit}_only_bboxes.jpg")
        cv2.imwrite(save_path, img_vis_only_bboxes)

    # visualize ground-truth co-attention points on the image
    coatt_level_gt = result_sample['coatt_level_gt'][0]
    coatt_level_gt_flag = torch.sum(coatt_level_gt == 1, dim=-1) > 1
    coatt_level_gt = coatt_level_gt[coatt_level_gt_flag]
    coatt_hm_gt_pk_cd = result_sample['coatt_hm_gt_pk_cd'][0][coatt_level_gt_flag]
    img_vis_gt = img.copy()

    for grp_idx, pk_cd in enumerate(coatt_hm_gt_pk_cd):
        pk_x = int(pk_cd[0].item() * img_w)
        pk_y = int(pk_cd[1].item() * img_h)

        grp_color = group_colors[grp_idx % len(group_colors)]

        # plot the ground-truth co-attention point
        cv2.circle(img_vis_gt, (pk_x, pk_y), 10, grp_color, -1)  # colored circle for ground-truth co-attention points
        
        # plot head boxes with high co-attention level for the group
        coatt_level_gt_grp = coatt_level_gt[grp_idx]
        print(f"GT Group {grp_idx}: Members {coatt_level_gt_grp}")
        for head_idx in range(coatt_level_gt_grp.shape[0]):
            if coatt_level_gt_grp[head_idx] == 1:
                head_box = head_boxes[head_idx]
                x1 = int(head_box[0].item() * img_w)
                y1 = int(head_box[1].item() * img_h)
                x2 = int(head_box[2].item() * img_w)
                y2 = int(head_box[3].item() * img_h)

                # change color based on group index
                cv2.rectangle(img_vis_gt, (x1, y1), (x2, y2), grp_color, 2)  # colored box for heads with high co-attention level
        
    # draw GT gaze target points if available
    gaze_pts_gt = result_sample['gaze_pts_gt'][0]
    for head_ibx in range(head_boxes.shape[0]):
        head_box = head_boxes[head_ibx]
        if torch.sum(head_box) == 0:
            continue  # skip if head box is not valid
        x1 = int(head_box[0].item() * img_w)
        y1 = int(head_box[1].item() * img_h)
        x2 = int(head_box[2].item() * img_w)
        y2 = int(head_box[3].item() * img_h)

        gaze_pts_gt_head = gaze_pts_gt[head_ibx]
        if torch.sum(gaze_pts_gt_head) == 0:
            continue  # skip if gaze point is not valid
        gaze_x = int(gaze_pts_gt_head[0].item() * img_w)
        gaze_y = int(gaze_pts_gt_head[1].item() * img_h)

        # plot the gaze target arrow
        start = ((x1 + x2) // 2, (y1 + y2) // 2)
        end = (gaze_x, gaze_y)
        # cv2.arrowedLine(img_vis_gt, start, end, (255, 255, 255), 2, cv2.LINE_AA, 0, 0.1)

    # save the visualization
    save_path = os.path.join(save_dir_data, f"gt.jpg")
    cv2.imwrite(save_path, img_vis_gt)