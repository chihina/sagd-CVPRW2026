import os
import glob
import json
import pandas as pd

def highlight_best(s, mode='max'):
    s_numeric = pd.to_numeric(s.str.replace(r'\\red\{(.*?)\}', r'\1', regex=True), errors='coerce')
    
    if mode == 'min':
        is_target = s_numeric == s_numeric.min()
    else:
        is_target = s_numeric == s_numeric.max()

    result = s.copy()
    for i, should_highlight in enumerate(is_target):
        if should_highlight:
            if not result.iloc[i].startswith(r'\red{'):
                result.iloc[i] = f"\\red{{{result.iloc[i]}}}"
    
    styles = ['background-color: yellow' if v else '' for v in is_target]
    
    return styles

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
    model_name = '09-12-36_combined_social_COA_False_bce_SOC_True_hm_coef_iter_temp_2-3_1_0.0001_FRZ_IT_VE'
    model_names.append(model_name)
    model_names_omit.append('MTGS-PP~\cite{DBLP:conf/nips/GuptaTFVO24}')

    model_name = '09-12-36_combined_social_COA_False_bce_SOC_True_hm_coef_iter_temp_2-3_1_0.0001_FRZ_IT_VE'
    model_names.append(model_name)
    model_names_omit.append('MTGS-Soc.~\cite{DBLP:conf/nips/GuptaTFVO24}')

    model_name = '11-39-06_gazelle_1e-06'
    model_names.append(model_name)
    model_names_omit.append('Gaze-LLE-PP~\cite{DBLP:conf/cvpr/RyanB0BHR25}')

    model_name = "17-23-01_combined_social_COA_True_True_bce_SOC_True_hm_coef_iter_temp_2-3_1_1e-05_w_gaze_vec_FRZ_IT_VE"
    model_names.append(model_name)
    model_names_omit.append('Ours')

    model_name = "08-19-48_combined_social_COA_True_True_bce_SOC_True_hm_coef_iter_temp_2-3_1_w_gaze_vec_FRZ_IT_VE_1e-06"
    model_names.append(model_name)
    model_names_omit.append('Ours (1e-6)')

    model_name = "16-39-06_combined_social_COA_True_True_bce_SOC_False_hm_coef_iter_temp_2-3_1_w_gaze_vec_FRZ_IT_VE_1e-05"
    model_names.append(model_name)
    model_names_omit.append('Ours (w/o social loss)')

    model_name = "16-43-15_combined_social_COA_True_True_bce_SOC_True_hm_coef_temp_2-3_w_gaze_vec_FRZ_IT_VE_1e-05"
    model_names.append(model_name)
    model_names_omit.append('Ours (w/o iteration)')


elif dataset_type == 'videoattentiontarget':
    model_name = '09-12-36_combined_social_COA_False_bce_SOC_True_hm_coef_iter_temp_2-3_1_0.0001_FRZ_IT_VE'
    model_names.append(model_name)
    model_names_omit.append('MTGS-PP~\cite{DBLP:conf/nips/GuptaTFVO24}')

    model_name = '09-12-36_combined_social_COA_False_bce_SOC_True_hm_coef_iter_temp_2-3_1_0.0001_FRZ_IT_VE'
    model_names.append(model_name)
    model_names_omit.append('MTGS-Soc.~\cite{DBLP:conf/nips/GuptaTFVO24}')

    model_name = '11-39-06_gazelle_1e-06'
    model_names.append(model_name)
    model_names_omit.append('Gaze-LLE-PP~\cite{DBLP:conf/cvpr/RyanB0BHR25}')

    model_name = "17-23-01_combined_social_COA_True_True_bce_SOC_True_hm_coef_iter_temp_2-3_1_1e-05_w_gaze_vec_FRZ_IT_VE"
    model_names.append(model_name)
    model_names_omit.append('Ours')

elif dataset_type == 'childplay':
    model_name = '09-12-36_combined_social_COA_False_bce_SOC_True_hm_coef_iter_temp_2-3_1_0.0001_FRZ_IT_VE'
    model_names.append(model_name)
    model_names_omit.append('MTGS-PP~\cite{DBLP:conf/nips/GuptaTFVO24}')

    model_name = '09-12-36_combined_social_COA_False_bce_SOC_True_hm_coef_iter_temp_2-3_1_0.0001_FRZ_IT_VE'
    model_names.append(model_name)
    model_names_omit.append('MTGS-Soc.~\cite{DBLP:conf/nips/GuptaTFVO24}')

    model_name = '11-39-06_gazelle_1e-06'
    model_names.append(model_name)
    model_names_omit.append('Gaze-LLE-PP~\cite{DBLP:conf/cvpr/RyanB0BHR25}')

    model_name = "17-23-01_combined_social_COA_True_True_bce_SOC_True_hm_coef_iter_temp_2-3_1_1e-05_w_gaze_vec_FRZ_IT_VE"
    model_names.append(model_name)
    model_names_omit.append('Ours')

# create save directory
save_dir = os.path.join('analysis', 'excel', dataset_type, f'{os.path.basename(__file__)[:-3]}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# set parameters for both models
use_group_iou_thrs = [0.5, 0.75, 1.0]
use_dist_thrs = [0.05, 0.1, 100.0]

use_coatt_conf_thrs_ap = [0.1, 0.3, 0.5, 0.7, 0.9]
use_coatt_conf_thrs_ap_mtgs_pp = [0.05, 0.1, 0.15, 0.2]
use_coatt_conf_thrs_ap_mtgs_sp = [1.0, 1.1, 1.2, 1.3, 1.4]

use_coatt_conf_thrs_cost = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
use_coatt_conf_thrs_cost_mtgs_pp = [0.05, 0.1, 0.15, 0.2]
use_coatt_conf_thrs_cost_mtgs_sp = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

# prepare save columns
save_cols = {}

# save Group-AP
save_cols_ap = []
# for use_group_iou_thr in use_group_iou_thrs:
    # for use_coatt_conf_thr in use_coatt_conf_thrs_ap:
        # save_cols_ap.append(f'{use_group_iou_thr}_{use_coatt_conf_thr}')
# add maximum AP for each group IoU threshold
# for use_group_iou_thr in use_group_iou_thrs:
    # save_cols_ap.append(f'{use_group_iou_thr}_min')
for use_group_iou_thr in use_group_iou_thrs:
    for use_dist_thr in use_dist_thrs:
        save_cols_ap.append(f'{use_group_iou_thr}_{use_dist_thr}')
save_cols['ap'] = save_cols_ap

# save Group-Dist
save_cols_dist = []
for use_group_iou_thr in use_group_iou_thrs:
    save_cols_dist.append(f'{use_group_iou_thr}')
# for use_coatt_conf_thr in use_coatt_conf_thrs_cost:
    # save_cols_dist.append(f'{use_coatt_conf_thr}')
# add minimum Dist for each group IoU threshold
# for use_group_iou_thr in use_group_iou_thrs:
    # save_cols_dist.append(f'{use_group_iou_thr}_min')
# save_cols['dist'] = save_cols_dist

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
    print(f"Results for {model_name} ({model_name_omit})")

    # replace NaN with 10000
    for key in results:
        if isinstance(results[key], float) and (results[key] != results[key]):
            results[key] = 10000.0
    
    vals_ap = []
    vals_dist = []
    vals_cost = []

    if 'MTGS-Soc.' in model_name_omit:
        model_metric_ap = 'coatt_ap_pairs'
        model_metric_dist = 'coatt_dist_pairs'
        # model_metric_dist = 'coatt_cost_dist_sp'
        # model_metric_cost = 'coatt_cost_sp'
        model_metric_cost = 'coatt_cost_iou_pairs'
        use_coatt_conf_thrs_ap_use = use_coatt_conf_thrs_ap_mtgs_sp
        use_coatt_conf_thrs_dist_use = use_coatt_conf_thrs_ap_mtgs_sp
        use_coatt_conf_thrs_cost_use = use_coatt_conf_thrs_cost_mtgs_sp
    elif 'MTGS-PP' in model_name_omit or 'Gaze-LLE-PP' in model_name_omit:
        model_metric_ap = 'coatt_ap_pp'
        model_metric_dist = 'coatt_dist_pp'
        # model_metric_dist = 'coatt_cost_dist_pp'
        # model_metric_cost = 'coatt_cost_pp'
        model_metric_cost = 'coatt_cost_iou_pp'
        use_coatt_conf_thrs_ap_use = use_coatt_conf_thrs_ap_mtgs_pp
        use_coatt_conf_thrs_dist_use = use_coatt_conf_thrs_ap_mtgs_pp
        use_coatt_conf_thrs_cost_use = use_coatt_conf_thrs_cost_mtgs_pp
    elif 'Ours' in model_name_omit:
        model_metric_ap = 'coatt_ap_grp'
        model_metric_dist = 'coatt_dist_grp'
        # model_metric_dist = 'coatt_cost_dist_grp'
        # model_metric_cost = 'coatt_cost_grp'
        model_metric_cost = 'coatt_cost_iou_grp'
        use_coatt_conf_thrs_ap_use = use_coatt_conf_thrs_ap
        use_coatt_conf_thrs_dist_use = use_coatt_conf_thrs_ap
        use_coatt_conf_thrs_cost_use = use_coatt_conf_thrs_cost


    """
    for use_group_iou_thr in use_group_iou_thrs:
        for use_coatt_conf_thr in use_coatt_conf_thrs_ap_use:
            save_cols_ours_ap = f'{model_metric_dist}_{use_group_iou_thr}_{use_coatt_conf_thr}'
            for col, val in results.items():
                if col == save_cols_ours_ap:
                    vals_dist.append(val)
    """

    """
    for use_coatt_conf_thr in use_coatt_conf_thrs_cost_use:
        save_cols_ours_dist = f'{model_metric_dist}_{use_coatt_conf_thr}'
        for col, val in results.items():
            if col == save_cols_ours_dist:
                vals_dist.append(val)
    """
    
    
    # Get Dist values
    '''
    for use_group_iou_thr in use_group_iou_thrs:
        vals_dist_child = []
        for use_dist_thr in use_dist_thrs:
            vals_dist_child.extend(results[f'{model_metric_dist}_{use_group_iou_thr}_{use_dist_thr}_{thr}'] for thr in use_coatt_conf_thrs_dist_use)
        vals_dist.append(min(vals_dist_child))
    '''
    # min_dist = min([results[f'{model_metric_dist}_{thr}'] for thr in use_coatt_conf_thrs_cost_use])
    # vals_dist.append(min_dist)

    # Get AP values
    for use_group_iou_thr in use_group_iou_thrs:
        for use_dist_thr in use_dist_thrs:
            max_ap = max([results[f'{model_metric_ap}_{use_group_iou_thr}_{use_dist_thr}_{thr}'] for thr in use_coatt_conf_thrs_ap_use])
            max_ap_thr = use_coatt_conf_thrs_ap_use[[results[f'{model_metric_ap}_{use_group_iou_thr}_{use_dist_thr}_{thr}'] for thr in use_coatt_conf_thrs_ap_use].index(max_ap)]
            print(f"AP for {model_name_omit} at IoU {use_group_iou_thr} and Dist {use_dist_thr}: {max_ap:.4f} (at co-attention confidence threshold {max_ap_thr})")
            vals_ap.append(max_ap)

        # min_dist = min([results[f'{model_metric_dist}_{use_group_iou_thr}_{thr}'] for thr in use_coatt_conf_thrs_ap_use])
        # vals_dist.append(min_dist)

    # for use_coatt_conf_thr in use_coatt_conf_thrs_cost:
        # save_cols_ours_cost = f'{model_metric_cost}_{use_coatt_conf_thr}'
        # for col, val in results.items():
            # if col == save_cols_ours_cost:
                # vals_cost.append(val)

    # Get Cost values  
    # '''      
    max_cost = max([results[f'{model_metric_cost}_{thr}'] for thr in use_coatt_conf_thrs_cost_use])
    vals_cost.append(max_cost)
    # '''      

    save_vals_ap.append(vals_ap)
    save_vals_dist.append(vals_dist)
    save_vals_cost.append(vals_cost)

save_vals['ap'] = save_vals_ap
save_vals['cost'] = save_vals_cost
# save_vals['dist'] = save_vals_dist

# create dataframe and save to csv
evaluate_metrics = ['ap', 'cost']
for metric in evaluate_metrics:
    print(f"Processing metric: {metric}")
    save_vals_met = save_vals[metric]
    save_cols_met = save_cols[metric]
    print(f"Save columns: {save_cols_met}")
    print(f"Save values: {save_vals_met}")

    # generate dataframe
    df = pd.DataFrame(save_vals_met, columns=save_cols_met, index=model_names_omit)

    # change the range from 0-1 to 0-100 for better readability
    df = df * 100

    # change the data type to float
    df = df.astype(float)

    # round to 1 decimal place
    df = df.round(1)
    
    # save to excel
    df_display = df.copy().astype(str)
    if metric in ['dist']:
        for col in df.columns:
            mask = df[col] == df[col].min()
            for idx in df[mask].index:
                df_display.loc[idx, col] = f"\\red{{{df.loc[idx, col]}}}"
    else:  # 'ap', 'cost'
        for col in df.columns:
            mask = df[col] == df[col].max()
            for idx in df[mask].index:
                df_display.loc[idx, col] = f"\\red{{{df.loc[idx, col]}}}"
    
    # save to excel
    save_path = os.path.join(save_dir, f'comparison_{dataset_type}_{metric}.xlsx')
    df.to_excel(save_path)

    # save to csv
    save_path = os.path.join(save_dir, f'comparison_{dataset_type}_{metric}.csv')
    df.to_csv(save_path)

    # highlight best results and save to excel
    if metric in ['dist']:
        df_style = df_display.style.apply(highlight_best, axis=0, mode='min')
    elif metric in ['ap', 'cost']:
        df_style = df_display.style.apply(highlight_best, axis=0, mode='max')
    
    # save highlighted excel
    highlight_save_path = os.path.join(save_dir, f'comparison_{dataset_type}_{metric}_highlighted.xlsx')
    df_style.to_excel(highlight_save_path)