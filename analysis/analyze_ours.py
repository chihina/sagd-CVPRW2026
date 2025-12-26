import os
import glob
import json
import pandas as pd

# set result directory
result_dir = os.path.join('checkpoints')

# create save directory
save_dir = os.path.join('analysis', 'excel', f'{os.path.basename(__file__)[:-3]}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# set models
model_name = '10-24-42_COA_True_SOC_True'

# set dataset type
dataset_type = 'videocoatt'

result_json_path = glob.glob(os.path.join(result_dir, '*', '*', model_name, dataset_type, '*.json'))
with open(result_json_path[0], 'r') as f:
    results = json.load(f)
print(f"Results for {model_name}: {results}")

# set thresholds to analyze
group_iou_thresholds = [0.75, 1.0]
coatt_conf_thresholds = [0.3, 0.5, 0.7]

# save excel files for each Group-IoU threshold
for group_iou_thr in group_iou_thresholds:
    cols = []
    vals = []
    for coatt_conf_thr in coatt_conf_thresholds:
        col_name = f'coatt_ap_sim_grp_{coatt_conf_thr}_{group_iou_thr}'
        cols.append(f'{coatt_conf_thr}')
        val = results[col_name]
        vals.append(val)

    df_group_iou = pd.DataFrame([vals], columns=cols)
    save_path = os.path.join(save_dir, f'group_iou_{group_iou_thr}.xlsx')
    df_group_iou.to_excel(save_path)

# save excel files for each Co-Attention confidence threshold
for coatt_conf_thr in coatt_conf_thresholds:
    cols = []
    vals = []
    for group_iou_thr in group_iou_thresholds:
        col_name = f'coatt_ap_sim_grp_{coatt_conf_thr}_{group_iou_thr}'
        cols.append(f'{group_iou_thr}')
        val = results[col_name]
        vals.append(val)

    df_coatt_conf = pd.DataFrame([vals], columns=cols)
    save_path = os.path.join(save_dir, f'coatt_conf_{coatt_conf_thr}.xlsx')
    df_coatt_conf.to_excel(save_path)