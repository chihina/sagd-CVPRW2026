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
dataset_type = 'videocoatt'
# dataset_type = 'videoattentiontarget'
# dataset_type = 'childplay'

if dataset_type == 'videocoatt':
    dataset_dir = os.path.join('data', 'VideoCoAtt_Dataset', 'images_nk')
    coatt_level_thresh = 0.79
elif dataset_type == 'videoattentiontarget':
    dataset_dir = os.path.join('data', 'videoattentiontarget', 'images')
    coatt_level_thresh = 0.35
elif dataset_type == 'childplay':
    dataset_dir = os.path.join('data', 'ChildPlay-gaze', 'images')
    coatt_level_thresh = 0.1
else:
    raise ValueError(f"Unknown dataset type: {dataset_type}")

model_name = "17-23-01_combined_social_COA_True_True_bce_SOC_True_hm_coef_iter_temp_2-3_1_1e-05_w_gaze_vec_FRZ_IT_VE"
model_names.append(model_name)
model_name_omit = "Ours"
model_names_omit.append(model_name_omit)

# create save directory
save_dir = os.path.join('analysis', 'visualization', dataset_type, f'{os.path.basename(__file__)[:-3]}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_data_ious = []
save_data_dists = []

result_pickle_paths = glob.glob(os.path.join(result_dir, '*', '*', model_name, dataset_type, 'test-predictions.pickle'))
result_pickle_path = result_pickle_paths[0]
results = fast_pickle_load(result_pickle_path)

save_names = []
for idx, result in enumerate(results):
    # retrieve sample path
    paths = result['path']
    sample_path = paths[len(paths)//2][0]  # middle frame path
    save_names.append(sample_path)

    # retrieve group IoUs
    group_iou_grp = result['group_iou_grp'].item()

    # retrieve group distances
    group_dist_grp = result['group_dist_grp'].item()

    save_data_ious_result = []
    save_data_dists_result = []
    save_data_ious_result.append(group_iou_grp)
    save_data_dists_result.append(group_dist_grp)
    save_data_ious.append(save_data_ious_result)
    save_data_dists.append(save_data_dists_result)

print(results[0].keys())

print(f'Load {len(results)} results for each model.')

# generate dataframes
df_ious = pd.DataFrame(np.array(save_data_ious).reshape(-1, len(model_names)), columns=model_names_omit, index=save_names)
df_dists = pd.DataFrame(np.array(save_data_dists).reshape(-1, len(model_names)), columns=model_names_omit, index=save_names)
print("IoU DataFrame:")
print(df_ious)

iou_diff_threshold = 0.8
dist_diff_threshold = 0.2

# initialize all indices as selected
selected_indices = df_ious[df_ious['Ours'] > iou_diff_threshold].index.tolist()
print(f"Number of selected samples: {len(selected_indices)}")

df_selected_ious = df_ious.loc[selected_indices]
df_selected_dists = df_dists.loc[selected_indices]
print("Selected IoU DataFrame:")
print(df_selected_ious)
print("Selected Distance DataFrame:")
print(df_selected_dists)

group_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128)]

# randomly select 10 samples for visualization if there are more than 10 selected samples
np.random.seed(777)  # for reproducibility
if len(selected_indices) > 10:
    selected_indices = np.random.choice(selected_indices, 10, replace=False).tolist()
    print(f"Randomly selected {len(selected_indices)} samples for visualization.")

for data_id in selected_indices:
    if dataset_type == 'videoattentiontarget':
        idx_top = data_id.split('/')[0].replace('_', ' ')
        idx = os.path.join(idx_top, *data_id.split('/')[1:])
    elif dataset_type == 'videocoatt':
        idx = data_id

    save_dir_data = os.path.join(save_dir, data_id.replace('/', '_'))
    if not os.path.exists(save_dir_data):
        os.makedirs(save_dir_data)
    
    print(f"Visualizing {data_id}...")

    # visualize results for each model
    result_sample = next((res for res in results if res['path'][len(results[0]['path'])//2][0] == data_id), None)

    # read images 
    img_path = os.path.join(dataset_dir, idx)
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    img_vis = img.copy()
    img_vis_init = img.copy()
    img_vis_init_with_values = img.copy()
    img_vis_final = img.copy()
    img_vis_final_with_values = img.copy()

    # save the original image
    save_path = os.path.join(save_dir_data, f"img.jpg")
    cv2.imwrite(save_path, img_vis) 

    # retrieve predicted co-attention points
    coatt_level_pred_final = torch.sigmoid(result_sample['coatt_level_pred'][0])
    coatt_level_final_flag = torch.sum(coatt_level_pred_final > coatt_level_thresh, dim=-1) > 1
    coatt_level_pred_final = coatt_level_pred_final[coatt_level_final_flag]
    coatt_hm_pred_pk_cd_final = result_sample['coatt_hm_pred_pk_cd'][0][coatt_level_final_flag]

    coatt_level_pred_init = torch.sigmoid(result_sample['coatt_level_pred_all'][0, 0])
    coatt_level_init_flag = torch.sum(coatt_level_pred_init > coatt_level_thresh, dim=-1) > 1
    coatt_level_pred_init = coatt_level_pred_init[coatt_level_init_flag]
    coatt_hm_pred_pk_cd_init = result_sample['coatt_hm_pred_pk_cd_all'][0, 0][coatt_level_init_flag]

    head_boxes = result_sample['head_bboxes'][0]

    # visualize initial co-attention points and head boxes with high co-attention level for the group
    for grp_idx, pk_cd in enumerate(coatt_hm_pred_pk_cd_init):
        pk_x = int(pk_cd[0].item() * img_w)
        pk_y = int(pk_cd[1].item() * img_h)
        grp_color = group_colors[grp_idx % len(group_colors)]
        grp_member = 0

        # plot head boxes with high co-attention level for the group
        coatt_level_pred_grp = coatt_level_pred_final[grp_idx]
        for head_idx in range(coatt_level_pred_grp.shape[0]):
            if coatt_level_pred_grp[head_idx] > coatt_level_thresh:
                head_box = head_boxes[head_idx]
                if torch.sum(head_box) == 0:
                    continue  # skip if head box is not valid
                grp_member += 1
            
        if grp_member < 2:
            continue  # skip if there are less than 2 members in the group

        # plot head boxes with high co-attention level for the group
        coatt_level_init_grp = coatt_level_pred_init[grp_idx]
        print(f"Initial Group {grp_idx}: Members {coatt_level_init_grp}")
        for head_idx in range(coatt_level_init_grp.shape[0]):
            if coatt_level_init_grp[head_idx] > coatt_level_thresh:
                head_box = head_boxes[head_idx]
                x1 = int(head_box[0].item() * img_w)
                y1 = int(head_box[1].item() * img_h)
                x2 = int(head_box[2].item() * img_w)
                y2 = int(head_box[3].item() * img_h)

                # change color based on group index
                cv2.rectangle(img_vis_init, (x1, y1), (x2, y2), grp_color, 2)  # colored box for heads with high co-attention level
                cv2.rectangle(img_vis_init_with_values, (x1, y1), (x2, y2), grp_color, 2)  # colored box for heads with high co-attention level
                cv2.putText(img_vis_init_with_values, f"{coatt_level_init_grp[head_idx]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grp_color, 1)  # add co

        # plot the initial co-attention point
        cv2.circle(img_vis_init, (pk_x, pk_y), 10, grp_color, -1)  # colored circle for initial co-attention points
        cv2.circle(img_vis_init_with_values, (pk_x, pk_y), 10, grp_color, -1)  # colored circle for initial co-attention points

    # visualize final co-attention points and head boxes with high co-attention level for the group
    for grp_idx, pk_cd in enumerate(coatt_hm_pred_pk_cd_final):
        pk_x = int(pk_cd[0].item() * img_w)
        pk_y = int(pk_cd[1].item() * img_h)
        grp_color = group_colors[grp_idx % len(group_colors)]
        grp_member = 0

        # plot head boxes with high co-attention level for the group
        coatt_level_pred_grp = coatt_level_pred_final[grp_idx]
        for head_idx in range(coatt_level_pred_grp.shape[0]):
            if coatt_level_pred_grp[head_idx] > coatt_level_thresh:
                head_box = head_boxes[head_idx]
                if torch.sum(head_box) == 0:
                    continue  # skip if head box is not valid
                grp_member += 1
        
        if grp_member < 2:
            continue  # skip if there are less than 2 members in the group

        # plot head boxes with high co-attention level for the group
        coatt_level_final_grp = coatt_level_pred_final[grp_idx]
        print(f"Final Group {grp_idx}: Members {coatt_level_final_grp}")
        for head_idx in range(coatt_level_final_grp.shape[0]):
            if coatt_level_final_grp[head_idx] > coatt_level_thresh:
                head_box = head_boxes[head_idx]
                x1 = int(head_box[0].item() * img_w)
                y1 = int(head_box[1].item() * img_h)
                x2 = int(head_box[2].item() * img_w)
                y2 = int(head_box[3].item() * img_h)

                # change color based on group index
                cv2.rectangle(img_vis_final, (x1, y1), (x2, y2), grp_color, 2)  # colored box for heads with high co-attention level
                cv2.rectangle(img_vis_final_with_values, (x1, y1), (x2, y2), grp_color, 2)  # colored box for heads with high co-attention level
                cv2.putText(img_vis_final_with_values, f"{coatt_level_final_grp[head_idx]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grp_color, 1)  # add co

        # plot the final co-attention point
        cv2.circle(img_vis_final, (pk_x, pk_y), 10, grp_color, -1)  # colored circle for final co-attention points
        cv2.circle(img_vis_final_with_values, (pk_x, pk_y), 10, grp_color, -1)  # colored circle for final co-attention points

    # save the visualization
    save_path = os.path.join(save_dir_data, f"{model_name_omit}_init.jpg")
    cv2.imwrite(save_path, img_vis_init)
    save_path = os.path.join(save_dir_data, f"{model_name_omit}_init_with_values.jpg")
    cv2.imwrite(save_path, img_vis_init_with_values)
    save_path = os.path.join(save_dir_data, f"{model_name_omit}_final.jpg")
    cv2.imwrite(save_path, img_vis_final)
    save_path = os.path.join(save_dir_data, f"{model_name_omit}_final_with_values.jpg")
    cv2.imwrite(save_path, img_vis_final_with_values)

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

    # save the visualization
    save_path = os.path.join(save_dir_data, f"gt.jpg")
    cv2.imwrite(save_path, img_vis_gt)