import pickle
import torch
import random
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score
from tqdm import tqdm
import itertools
from src.metrics import GFTestDistance
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def compute(results, dataset=None, shuffle=False, lah_csv_path=False, laeo_csv_path=False, thr=0.6):
    
    gf_metrics = GFTestDistance()
    print('Computing metrics...')
    if shuffle:
        random.shuffle(results)
    # iterate over batches
    lah_gt_all = []; lah_pred_all = []; laeo_gt_all = []; laeo_pred_all = []; coatt_gt_all = []; coatt_pred_all = []; distances = []; avg_distances = []; inout_gt_all = []; inout_pred_all = []; mask_all = []
    
    head_bboxes_all = []; gp_pred_all = []; laeo_ids_all = []; paths = []; pids = []
    for batch in tqdm(results):
        
        # filter based on dataset
        if dataset!=None:
            if batch['dataset'][0]!=dataset:
                continue
                
        # get distance score
        if batch['dataset'][0]=='gazefollow':
            test_dist_to_avg, _, test_min_dist = gf_metrics(batch['gp_pred'].cpu(), batch["gp_gt"].cpu())
            avg_distances.append(test_dist_to_avg.unsqueeze(0))
            distances.append(test_min_dist.unsqueeze(0))
        else:
            dist = (batch['gp_pred'] - batch['gp_gt']).norm(2, dim=-1)
            distances.append(dist[batch['inout_gt']==1].cpu())
            
        # get inout score
        if batch['dataset'][0] in ['videoattentiontarget', 'childplay']:
            mask = batch['inout_gt']!=-1
            inout_gt_all.append(batch['inout_gt'][mask].cpu())
            inout_pred_all.append(batch['inout_pred'][mask].cpu())
        
    # Distance metric
    if len(avg_distances)>0:
        avg_distances = torch.cat(avg_distances)
        print('Avg Dist: ', avg_distances.mean())
    if len(distances)>0:
        distances = torch.cat(distances)
        print('Dist: ', distances.mean())
        
    # Inout metric
    if len(inout_gt_all)>0:
        inout_gt_all = torch.cat(inout_gt_all); inout_pred_all = torch.cat(inout_pred_all)
        print('In-out AP: ', average_precision_score(inout_gt_all.float(), inout_pred_all.float()))
        
    
if __name__=='__main__':
    results_path = '/idiap/temp/afarkhondeh/code/ai4autism/gaze-interact/experiments/2024-05-15/19-00-30/test-predictions.pickle'
    print('Loading data...')
    with open(results_path, 'rb') as fp:
        results = pickle.load(fp)
        
#     print('GazeFollow')
#     compute(results, thr=0.5)
    print('LAEO')
    compute(results, dataset='laeo', thr=0.5)
    print('CoAtt')
    compute(results, dataset='coatt', thr=0.5)
    print('ChildPlay')
    compute(results, dataset='childplay', thr=0.5)
    print('VAT')
    compute(results, dataset='videoattentiontarget', thr=0.5)