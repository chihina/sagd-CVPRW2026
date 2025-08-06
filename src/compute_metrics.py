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
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


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

        batch_size, num_people = batch['head_bboxes'].shape[:2]
        pair_indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
        
        
        # save head_bboxes, gaze predictions and laeo ids
        if lah_csv_path or laeo_csv_path:
            head_bboxes_all.extend(batch['head_bboxes'].cpu().numpy())
            gp_pred_all.extend(batch['gp_pred'].float().cpu().numpy())
            laeo_ids_all.extend(batch['laeo_ids'].cpu().numpy())
            paths.extend(batch['path'][0]*num_people)
            bpids = batch['pids'][0].cpu()
            pids.extend(bpids.numpy())
        

        # get lah results
        lah_gt = batch['lah_gt'].cpu()
        lah_pred = batch['lah_pred'].cpu()
        lah_pred_argmax = torch.zeros_like(lah_pred)
        lah_gt_metric = torch.zeros(batch_size, num_people).long() - 1
        lah_pred_metric = torch.zeros(batch_size, num_people)
        for bi in range(batch_size):
            for pi in range(num_people):
                if batch['dataset'][0]=='gazefollow':
                    io = 1
                else:    
                    io = batch['inout_gt'][bi][pi]==1
                if io==1:
                    valid_indices = torch.where((pair_indices[:, 1]==pi).int())[0]
                    if valid_indices.shape[0]>0:
                        if (lah_gt[bi][valid_indices]!=-1).sum()==0:
                            continue

                        max_val, max_idx = torch.max(lah_pred[bi][valid_indices], 0)
                        lah_pred_argmax[bi][valid_indices[max_idx]] = max_val

                        lah_gt_metric[bi][pi] = lah_gt[bi][valid_indices][lah_gt[bi][valid_indices]!=-1].sum()
                        gt_idx = torch.where(lah_gt[bi][valid_indices]==1)[0]
                        if len(gt_idx)>0:
                            lah_pred_metric[bi][pi] = lah_pred_argmax[bi][valid_indices][gt_idx]
                        else:
                            lah_pred_metric[bi][pi] = max_val
        mask = lah_gt_metric!=-1
        mask_all.append(mask[0])
        lah_gt_all.append(lah_gt_metric[0].cpu())
        lah_pred_all.append(lah_pred_metric[0].cpu())
                    
        # get laeo results
        laeo_gt = batch['laeo_gt'].cpu()
        mask = laeo_gt!=-1
        if mask.sum()>0:
            laeo_pred = batch['laeo_pred']
#             # infer LAEO score from LAH score
#             laeo_pred = torch.zeros_like(lah_pred)
#             for bi in range(batch_size):
#                 for pi, pair in enumerate(pair_indices):
#                     corr_idx = torch.where((pair_indices==pair[[1,0]]).prod(-1))[0]
#                     laeo_pred[bi][pi] = 2/((1/lah_pred[bi][pi])+(1/lah_pred[bi][corr_idx]).float())   # geometric mean of LAH scores
            laeo_pred_argmax = torch.zeros_like(laeo_pred)
            for bi in range(batch_size):
                for pi in range(num_people):
                    valid_indices = torch.where((pair_indices[:, 1]==pi).int() * (pair_indices[:, 0]!=0).int())[0]
                    if valid_indices.shape[0]>0:
                        max_val, max_idx = torch.max(laeo_pred[bi][valid_indices], 0)
                        laeo_pred_argmax[bi][valid_indices[max_idx]] = max_val
            laeo_gt = laeo_gt[mask]
            laeo_pred_argmax = laeo_pred_argmax[mask]
            if len(laeo_gt)>0:
                laeo_gt_all.append(laeo_gt.cpu())
                laeo_pred_all.append(laeo_pred_argmax.float().cpu())
        
        # get coatt results
        mask = batch['coatt_gt']!=-1
        batch_coatt_gt = batch['coatt_gt'][mask]
        batch_coatt_pred = batch['coatt_pred'][mask]
        if len(batch_coatt_gt)>0:
            coatt_gt_all.append(batch_coatt_gt.cpu())
            coatt_pred_all.append(batch_coatt_pred.float().cpu())
        
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
        
        
    # LAEO metrics
    print('-----------LAEO metrics-----------')
    if len(laeo_gt_all)>0:
        laeo_gt_all = torch.cat(laeo_gt_all); laeo_pred_all = torch.cat(laeo_pred_all)
        print('AP: ', average_precision_score(laeo_gt_all, laeo_pred_all))
        print('AUC: ', roc_auc_score(laeo_gt_all, laeo_pred_all))
        if laeo_csv_path:
            df = pd.DataFrame()
#             head_bboxes_all = np.concatenate(head_bboxes_all, 0)
#             gp_pred_all = np.concatenate(gp_pred_all, 0)
#             laeo_ids_all = np.concatenate(laeo_ids_all, 0)
#             df['xmin'] = head_bboxes_all[:, 0]
#             df['ymin'] = head_bboxes_all[:, 1]
#             df['xmax'] = head_bboxes_all[:, 2]
#             df['ymax'] = head_bboxes_all[:, 3]
#             df['gaze_x'] = gp_pred_all[:, 0]
#             df['gaze_y'] = gp_pred_all[:, 1]
#             df['laeo_id'] = laeo_ids_all
#             df['path'] = paths
            df['laeo_pred_logit'] = laeo_pred_all
            df['gt_laeo'] = laeo_gt_all
            df.to_csv(laeo_csv_path)
        
        laeo_pred_thr = laeo_pred_all>thr
        print('Prec: ', precision_score(laeo_gt_all, laeo_pred_thr))
        print('Recall: ', recall_score(laeo_gt_all, laeo_pred_thr))
        print('F1: ', f1_score(laeo_gt_all, laeo_pred_thr))
        
    # LAH metrics
    print('------------LAH metrics-----------')
    if len(lah_gt_all)>0:
        lah_gt_all = torch.cat(lah_gt_all); lah_pred_all = torch.cat(lah_pred_all); mask_all = torch.cat(mask_all)
        lah_gt_all = lah_gt_all[mask_all]; lah_pred_all = lah_pred_all[mask_all]; 
        if lah_gt_all.sum()<len(lah_gt_all):
            print('AP: ', average_precision_score(lah_gt_all, lah_pred_all))
            print('AUC: ', roc_auc_score(lah_gt_all, lah_pred_all))
    #         prec, recall, _ = precision_recall_curve(lah_gt_all, lah_pred_all)
    #         plt.plot(recall, prec)
    #         plt.show()
            if lah_csv_path:
                pids = np.array(pids, dtype=np.int32)[mask_all]; paths = np.array(paths)[mask_all]
                df = pd.DataFrame()
                df['looking_head'] = lah_pred_all
                df['path'] = paths
                df['pid'] = pids
                df.to_csv(lah_csv_path)

        lah_pred_thr = lah_pred_all>thr
        print('Prec: ', precision_score(lah_gt_all, lah_pred_thr))
        print('Recall: ', recall_score(lah_gt_all, lah_pred_thr))
        print('F1: ', f1_score(lah_gt_all, lah_pred_thr))

    # CoAtt metrics
    print('-----------CoAtt metrics----------')
    if len(coatt_gt_all)>0:
        coatt_gt_all = torch.cat(coatt_gt_all); coatt_pred_all = torch.cat(coatt_pred_all)
        print('AP: ', average_precision_score(coatt_gt_all, coatt_pred_all))
        print('AUC: ', roc_auc_score(coatt_gt_all, coatt_pred_all))
#         prec, recall, _ = precision_recall_curve(coatt_gt_all, coatt_pred_all)
#         plt.plot(recall, prec)
#         plt.show()
    
    
if __name__=='__main__':
    results_path = "/idiap/temp/agupta/gaze_interact/experiments/2025-05-16/15-42-42/test-predictions.pickle"
    print('Loading data...')
    with open(results_path, 'rb') as fp:
        # results = pickle.load(fp)
        results = CPU_Unpickler(fp).load()
        
# #     print('GazeFollow')
    compute(results, thr=0.5)
    # print('LAEO')
    # compute(results, dataset='laeo', thr=0.5)
    # print('CoAtt')
    # compute(results, dataset='coatt', thr=0.5)
    # print('ChildPlay')
    # compute(results, dataset='childplay', thr=0.5)
    # print('VAT')
    # compute(results, dataset='videoattentiontarget', thr=0.5)
