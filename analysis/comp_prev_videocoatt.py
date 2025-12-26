import os
import glob
import json
import pandas as pd

def highlight_best(s, mode='max'):
    if mode == 'min':
        is_target = s == s.min()
    else:
        is_target = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_target]

# set result directory
result_dir = os.path.join('checkpoints')

# create save directory
save_dir = os.path.join('analysis', 'excel', f'{os.path.basename(__file__)[:-3]}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# set models
model_names = []
model_names_omit = []


model_names.append('11-31-47_COA_True_SOC_True_coatt_hm_coef_iter_3')
model_names_omit.append('Ours')

# model_names.append('22-25-17_COA_True_SOC_True_coatt_hm_coef')
model_names.append('11-31-47_COA_True_SOC_True_coatt_hm_coef_iter_3')
model_names_omit.append('MTGS')

# set dataset type
# dataset_type = 'videocoatt'
dataset_type = 'videoattentiontarget'

# set parameters for both models
use_group_iou_thrs = [0.5, 0.75, 1.0]
use_coatt_conf_thrs_ap = [0.1, 0.3, 0.5, 0.7, 0.9]
use_coatt_conf_thrs_ap_mtgs = [0.05, 0.1, 0.15, 0.2]
use_coatt_conf_thrs_cost = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
use_coatt_conf_thrs_cost_mtgs = [0.05, 0.1, 0.15, 0.2]

save_cols = {}

# save Group-AP
save_cols_ap = []
# for use_group_iou_thr in use_group_iou_thrs:
    # for use_coatt_conf_thr in use_coatt_conf_thrs_ap:
        # save_cols_ap.append(f'{use_group_iou_thr}_{use_coatt_conf_thr}')
# add maximum AP for each group IoU threshold
for use_group_iou_thr in use_group_iou_thrs:
    save_cols_ap.append(f'{use_group_iou_thr}_max')
save_cols['ap'] = save_cols_ap

# save Group-Dist
save_cols_dist = []
# for use_group_iou_thr in use_group_iou_thrs:
    # for use_coatt_conf_thr in use_coatt_conf_thrs_ap:
        # save_cols_dist.append(f'{use_group_iou_thr}_{use_coatt_conf_thr}')
# for use_coatt_conf_thr in use_coatt_conf_thrs_cost:
    # save_cols_dist.append(f'{use_coatt_conf_thr}')
# add minimum Dist for each group IoU threshold
# for use_group_iou_thr in use_group_iou_thrs:
    # save_cols_dist.append(f'{use_group_iou_thr}_min')
save_cols_dist.append('min')
save_cols['dist'] = save_cols_dist

save_cols_cost = []
# for use_coatt_conf_thr in use_coatt_conf_thrs_cost:
    # save_cols_cost.append(f'{use_coatt_conf_thr}')
# add maximum cost for each confidence threshold
save_cols_cost.append('max')

save_cols['cost'] = save_cols_cost

save_vals = {}
save_vals_ap = []
save_vals_dist = []
save_vals_cost = []
for model_name, model_name_omit in zip(model_names, model_names_omit):
    result_json_path = glob.glob(os.path.join(result_dir, '*', '*', model_name, dataset_type, '*.json'))
    assert len(result_json_path) == 1, f"Expected one result file for {model_name}, found {len(result_json_path)}"
    with open(result_json_path[0], 'r') as f:
        results = json.load(f)
    # print(f"Results for {model_name}: {results}")

    vals_ap = []
    vals_dist = []
    vals_cost = []

    if model_name_omit == 'MTGS':
        model_metric_ap = 'coatt_ap_pp'
        # model_metric_dist = 'coatt_dist_pp'
        model_metric_dist = 'coatt_cost_dist_pp'
        # model_metric_cost = 'coatt_cost_pp'
        model_metric_cost = 'coatt_cost_iou_pp'
        use_coatt_conf_thrs_ap = use_coatt_conf_thrs_ap_mtgs
        use_coatt_conf_thrs_dist = use_coatt_conf_thrs_ap_mtgs
        use_coatt_conf_thrs_cost = use_coatt_conf_thrs_cost_mtgs
    elif model_name_omit == 'Ours':
        model_metric_ap = 'coatt_ap_grp'
        # model_metric_dist = 'coatt_dist_grp'
        model_metric_dist = 'coatt_cost_dist_grp'
        # model_metric_cost = 'coatt_cost_grp'
        model_metric_cost = 'coatt_cost_iou_grp'


    """
    for use_group_iou_thr in use_group_iou_thrs:
        for use_coatt_conf_thr in use_coatt_conf_thrs_ap:
            save_cols_ours_ap = f'{model_metric_dist}_{use_group_iou_thr}_{use_coatt_conf_thr}'
            for col, val in results.items():
                if col == save_cols_ours_ap:
                    vals_dist.append(val)
    """

    """
    for use_coatt_conf_thr in use_coatt_conf_thrs_cost:
        save_cols_ours_dist = f'{model_metric_dist}_{use_coatt_conf_thr}'
        for col, val in results.items():
            if col == save_cols_ours_dist:
                vals_dist.append(val)
    """

    
    min_dist = min([results[f'{model_metric_dist}_{thr}'] for thr in use_coatt_conf_thrs_cost])
    vals_dist.append(min_dist)

    for use_group_iou_thr in use_group_iou_thrs:
        # add the maximum AP for each group IoU threshold
        max_ap = max([results[f'{model_metric_ap}_{use_group_iou_thr}_{thr}'] for thr in use_coatt_conf_thrs_ap])
        vals_ap.append(max_ap)
        # min_dist = min([results[f'{model_metric_dist}_{use_group_iou_thr}_{thr}'] for thr in use_coatt_conf_thrs_ap])
        # vals_dist.append(min_dist)

    # for use_coatt_conf_thr in use_coatt_conf_thrs_cost:
    #     save_cols_ours_cost = f'{model_metric_cost}_{use_coatt_conf_thr}'
    #     for col, val in results.items():
    #         if col == save_cols_ours_cost:
    #             vals_cost.append(val)
        
    # add the maximum cost for each confidence threshold
    max_cost = max([results[f'{model_metric_cost}_{thr}'] for thr in use_coatt_conf_thrs_cost])
    vals_cost.append(max_cost)

    save_vals_ap.append(vals_ap)
    save_vals_dist.append(vals_dist)
    save_vals_cost.append(vals_cost)

save_vals['ap'] = save_vals_ap
save_vals['dist'] = save_vals_dist
save_vals['cost'] = save_vals_cost

# create dataframe and save to csv
evaluate_metrics = ['ap', 'dist', 'cost']
for metric in evaluate_metrics:
    print(f"Processing metric: {metric}")
    save_vals_met = save_vals[metric]
    save_cols_met = save_cols[metric]
    print(f"Save columns: {save_cols_met}")
    print(f"Save values: {save_vals_met}")
    df = pd.DataFrame(save_vals_met, columns=save_cols_met, index=model_names_omit)
    save_path = os.path.join(save_dir, f'comparison_{dataset_type}_{metric}.xlsx')
    df.to_excel(save_path)
    print(f"Saved results to {save_path}")
    if metric in ['dist']:
        df_style = df.style.apply(highlight_best, axis=0, mode='min')
    elif metric in ['ap', 'cost']:
        df_style = df.style.apply(highlight_best, axis=0, mode='max')
    else:
        assert False, f"Unsupported metric for highlighting: {metric}"
    
    highlight_save_path = os.path.join(save_dir, f'comparison_{dataset_type}_{metric}_highlighted.xlsx')
    df_style.to_excel(highlight_save_path)
    print(f"Saved highlighted results to {highlight_save_path}")