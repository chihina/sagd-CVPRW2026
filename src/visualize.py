import pickle
import torch
import torchvision
from torchvision.ops import nms
from tqdm import tqdm
import itertools
import numpy as np
import os
import cv2
import sys
import joblib
import time
import networkx as nx
import community as community

def get_root(batch):
    root_gf = '/home-local/nakatani/dataset/gazefollow_extended'
    root_coatt = '/home-local/nakatani/dataset/VideoCoAtt_Dataset'
    root_vat = '/home-local/nakatani/dataset/videoattentiontarget'
    root_laeo = '/home-local/nakatani/dataset/ucolaeodb/'
    root_childplay = '/home-local/nakatani/dataset/ChildPlay-gaze'

    root = ''
    if 'dataset' in batch.keys():
        if batch['dataset'][0]=='videoattentiontarget':
            root = os.path.join(root_vat, 'images')
        elif batch['dataset'][0]=='childplay':
            root = root_childplay
        elif batch['dataset'][0]=='laeo':
            root = os.path.join(root_laeo)
        elif batch['dataset'][0]=='coatt':
            root = os.path.join(root_coatt, 'images_nk')
        elif batch['dataset'][0]=='gazefollow':
            root = os.path.join(root_gf)

    return root
            
# auxiliary functions for drawing
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), 
           (0, 255, 255), (0, 255, 165), (255, 165, 0), (0, 165, 255), (255, 0, 165), (165, 255, 0), 
           (165, 255, 165), (255, 165, 165), (165, 165, 255), (165, 165, 0), (165, 85, 85), (85, 0, 85), (85, 85, 85),

           (255, 85, 85), (85, 255, 85), (85, 85, 255), (255, 255, 165), (165, 255, 255), (255, 165, 255),
           (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
           (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192),
           (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64)]

# get pixel coordinates for head bbox
def perc_to_pixel(head_bbox, frame_height, frame_width):
    
    head_bbox[0] *= frame_width
    head_bbox[1] *= frame_height
    head_bbox[2] *= frame_width
    head_bbox[3] *= frame_height
    
    return head_bbox

# function for visualizing gaze predictions
def visualize_gaze(batch, social_preds):
    
    root = get_root(batch)
    # path_list = batch['path'][0]
    # path = path_list[len(path_list)//2]
    path_list = batch['path']
    mid_idx = len(path_list)//2
    path = path_list[mid_idx][0]
    if batch['dataset'][0]=='videoattentiontarget':
        path = modify_path_vat(path)

    frame = cv2.imread(os.path.join(root, path))
    frame_height, frame_width = frame.shape[:2]

    # iterate over all people
    # print('batch keys:', batch.keys())
    num_people = len(batch['head_bboxes'][0])
    for i in range(1, num_people):

        # because of no initial padding
        if batch['dataset'][0]=='gazefollow':
            i -= 1

        # check if head bbox is valid
        head_bbox = batch['head_bboxes'][0][i].clone()
        if torch.sum(head_bbox) == 0:
            continue
        
        # draw head bbox
        head_bbox = perc_to_pixel(head_bbox, frame_height, frame_width).int().cpu().numpy()
        color = colors[i]
        thickness = 1
        # frame = cv2.rectangle(frame, head_bbox[:2], head_bbox[2:], color, thickness)

        # write pid
        org = (head_bbox[0], head_bbox[1] + 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color_text = (255, 255, 255)
        frame = cv2.putText(frame, str(i), org, font, fontScale, color_text, thickness, cv2.LINE_AA)

        # check if head inout prediction is over threshold
        head_inout_pred = batch['inout_pred'][0][i]
        head_inout_pred = torch.sigmoid(head_inout_pred)
        head_inout_gt = batch['inout_gt'][0][i]
        head_inout_thresh = 0.5
        # print(f'Head {i} inout pred:', head_inout_pred.item(), 'gt:', head_inout_gt.item())
        if head_inout_pred < head_inout_thresh:
            continue

        # draw gaze vector
        gaze_vec_pred = batch['gv_pred'][0][i]
        start = (head_bbox[:2] + head_bbox[2:])/2
        # gv_pred = gaze_vec_pred[0][i].clone().cpu().numpy()
        gv_pred = gaze_vec_pred.clone().cpu().numpy()
        end = start + gv_pred*100
        # frame = cv2.line(frame, start.astype(np.int16), end.astype(np.int16), color_text, thickness)

        # draw gaze point
        gaze_pt_pred = batch['gp_pred'][0][i]
        start = (head_bbox[:2] + head_bbox[2:])/2
        # gp_pred = gaze_pt_pred[0][i].clone().cpu().numpy()
        gp_pred = gaze_pt_pred.clone().cpu().numpy()
        gp_pred *= [frame_width, frame_height]
        frame = cv2.line(frame, start.astype(np.int16), gp_pred.astype(np.int16), color, thickness)
        radius = 10
        # frame = cv2.circle(frame, gp_pred.astype(np.int16), radius, color, thickness=-1)

        # drwa gaze point (GT)
        gaze_pt_gt = batch['gp_gt'][0][i]
        gp_gt = gaze_pt_gt.clone().cpu().numpy()
        gp_gt *= [frame_width, frame_height]
        # frame = cv2.line(frame, start.astype(np.int16), gp_gt.astype(np.int16), (0, 0, 0), thickness)
        radius = 10
        # frame = cv2.circle(frame, gp_gt.astype(np.int16), radius, (0, 0, 0), thickness=-1)

    # iterate over tasks
    # indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
    # lah_person = np.zeros(num_people, dtype=np.int16)-1
    # laeo_person = np.zeros(num_people, dtype=np.int16)-1
    # coatt_person = []
    # for i in range(num_people):
    #     coatt_person.append([])
    # social_persons = [lah_person, laeo_person, coatt_person]
    
    # thresholds = [0.6, 0.6, 0.4]
    # for t, task in enumerate(['lah', 'laeo', 'coatt']):
    #     social_pred = social_preds[t]
    #     social_person = social_persons[t]
    #     thres = thresholds[t]
    #     # iterate over pairs
    #     for pair_num, pair_score in enumerate(social_pred[0]):
    #         if pair_score > thres:
    #             pair_indices = indices[pair_num]
    #             if pair_indices[0]==0 or pair_indices[1]==0:
    #                 continue
    #             if task=='coatt':
    #                 social_person[pair_indices[0]].append(pair_indices[1].item())
    #                 social_person[pair_indices[1]].append(pair_indices[0].item())
    #             else:
    #                 if task=='laeo':
    #                     social_person[pair_indices[0]] = pair_indices[1].item()
    #                 social_person[pair_indices[1]] = pair_indices[0].item()

    # iterate over tasks
    # text_offset = 0
    # for t, task in enumerate(['lah', 'laeo', 'coatt']):
    #     social_person = social_persons[t]
    #     text_offset += 30
    #     # iterate over heads
    #     for i in range(num_people):
    #         color = colors[i]
    #         # write task over head bbox
    #         flag = 0
    #         if task=='coatt':
    #             if social_person[i]!=[]:
    #                 flag = 1
    #         else:
    #             if social_person[i]!=-1:
    #                 flag = 1
    #         if flag:
    #             social_person[i] = str(set(social_person[i]))
    #             head_bbox = batch['head_bboxes'][0][i].clone()
    #             head_bbox = perc_to_pixel(head_bbox, frame_height, frame_width).int().cpu().numpy()
    #             org = (head_bbox[0], head_bbox[3] + text_offset)
    #             frame = cv2.putText(frame, task+': '+str(social_person[i]), org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    return frame

def visualize_coatt(frame, batch):
    h, w = frame.shape[:2]
    coatt_hm = batch['coatt_hm_pred'][0]
    coatt_level = torch.sigmoid(batch['coatt_level_pred'][0])
    coatt_level_gt = batch['coatt_level_gt'][0]
    hm_pred = batch['hm_pred'][0, :, :, :]
    head_bboxes = batch['head_bboxes'][0]
    coatt_pred = batch['coatt_pred'][0]

    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    if batch['dataset'][0]=='videoattentiontarget':
        coatt_level_thresh = 0.2
    elif batch['dataset'][0]=='childplay':
        coatt_level_thresh = 0.2
    elif batch['dataset'][0]=='coatt':
        coatt_level_thresh = 0.3
    else:
        coatt_level_thresh = 0.2

    # """
    # get co-attention level using group detection
    for token_idx in range(coatt_level.shape[0]):
        coatt_level_pred = coatt_level[token_idx, :]
        coatt_hm_pred = coatt_hm[token_idx, :, :]
        # print(f'CoAtt Token {token_idx}', coatt_hm_pred)
        coatt_level_exist = torch.sum(coatt_level_pred > coatt_level_thresh) > 1
        # print(f'Token {token_idx} exist:', coatt_level_pred, coatt_level_exist)
        if coatt_level_exist:
            print(f'Token {token_idx}:', coatt_level_pred)
            for person_idx in range(coatt_level_pred.shape[0]):
                coatt_level_pred_person = coatt_level_pred[person_idx]
                head_bbox = head_bboxes[person_idx].clone()
                if torch.sum(head_bbox) == 0:
                    continue
                head_bbox = perc_to_pixel(head_bbox, h, w).int().cpu().numpy()
                color = colors[person_idx]
                thickness = 2
                frame = cv2.rectangle(frame, head_bbox[:2], head_bbox[2:], color, thickness)

                # plot the coatt level prediction per person
                org = (head_bbox[0], head_bbox[1] - 10)
                fin = (head_bbox[0], head_bbox[3] + 15)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                thickness = 1
                color_text = (255, 255, 255)
                frame = cv2.putText(frame, f'{coatt_level_pred_person:.2f}', fin, font, fontScale, color_text, thickness, cv2.LINE_AA)

            # generate co-attention heatmap
            coatt_mask = coatt_level_pred > coatt_level_thresh
            coatt_hm_pred = coatt_hm_pred * coatt_mask[:, None, None]
            coatt_hm_pred = torch.mean(coatt_hm_pred, 0)

            coatt_hm_pred = (coatt_hm_pred - torch.min(coatt_hm_pred.view(-1))) / (torch.max(coatt_hm_pred.view(-1)) - torch.min(coatt_hm_pred.view(-1)))
            coatt_hm_pred = cv2.resize(coatt_hm_pred.cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
            coatt_hm_pred = (coatt_hm_pred * 255).astype(np.uint8)
            coatt_hm_pred = cv2.applyColorMap(coatt_hm_pred, cv2.COLORMAP_JET)
            overlay_ratio = 0.3
            frame = cv2.addWeighted(frame, 1.0-overlay_ratio, coatt_hm_pred, overlay_ratio, 0)
    # """

    """
    # get co-attention groups using pairwise co-attention prediction
    num_people = head_bboxes.shape[0]
    indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
    edges = []
    for ind in range(indices.shape[0]):
        i, j = indices[ind]
        edges.append((i.item(), j.item(), {'weight': coatt_pred[ind].item()}))
    G = nx.Graph()
    G.add_edges_from(edges)
    # resolution = 1.5
    resolution = 1.25
    partition = community.best_partition(G, weight='weight', resolution=resolution)
    group_ids = list(set(partition.values()))
    node_ids = list(partition.keys())
    for g_id in group_ids:
        members = [node for node, comm_id in partition.items() if comm_id == g_id]
        if len(members) < 2:
            continue
        for person_idx in members:
            head_bbox = head_bboxes[person_idx].clone()
            head_bbox = perc_to_pixel(head_bbox, h, w).int().cpu().numpy()
            print(f'Group {g_id} Person {person_idx} bbox:', head_bbox)
            fin = (head_bbox[0], head_bbox[3] + 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            color_text = (255, 255, 255)
            frame = cv2.putText(frame, f'{1.00:.2f}', fin, font, fontScale, color_text, thickness, cv2.LINE_AA)
    """

    # """            
    # plot ground truth co-attention groups
    for token_idx in range(coatt_level.shape[0]):
        coatt_level_gt_token = coatt_level_gt[token_idx, :]
        if torch.sum(coatt_level_gt_token) > 1:
            # print(f'GT Token {token_idx}:', coatt_level_gt_token)
            for person_idx in range(coatt_level_gt_token.shape[0]):
                if coatt_level_gt_token[person_idx] == 1:
                    head_bbox = head_bboxes[person_idx].clone()
                    head_bbox = perc_to_pixel(head_bbox, h, w).int().cpu().numpy()
                    # print(f'Person {person_idx} bbox:', head_bbox)
                    color = (255, 255, 255)
                    thickness = 2
                    frame = cv2.rectangle(frame, head_bbox[:2], head_bbox[2:], color, thickness)
                    # print(f'GT Token {token_idx} Person {person_idx}')
    # """

    return frame

def modify_path_vat(path):
    data_name = path.split('/')[-3].replace('_', ' ')
    path_par = '/'.join(path.split('/')[-3:])
    path = os.path.join(data_name, path.split('/')[-2], path.split('/')[-1])

    return path

# main function
def compute(results, clip_name):
        
    print('Getting frames...')
    rel_batches = []; rel_paths = []
    for batch in tqdm(results):
        
        # filter based on clip name
        # t = len(batch['path'][0])
        t = len(batch['path'])
        middle_frame_idx = int(t/2)
        # path = batch['path'][0][middle_frame_idx]
        path = batch['path'][middle_frame_idx][0]

        # print('Path:', path)
        # if batch['dataset'][0]=='videoattentiontarget':
            # path = modify_path_vat(path)
        # print('Modified path:', path)

        cname = '/'.join(path.split('/')[:-1])
        if cname!=clip_name:
            continue
            
        rel_batches.append(batch)
        rel_paths.append(path)
    
    if len(rel_batches)==0:
        print('No relevant batches found for clip:', clip_name)
        return

    # read one frame
    root = get_root(batch)
    if batch['dataset'][0]=='videoattentiontarget':
        path = modify_path_vat(path)
    img_path = os.path.join(root, path)
    frame_tmp = cv2.imread(os.path.join(root, path))
    h, w = frame_tmp.shape[:2]
    
    # iterate over sorted frames
    print('Getting results...')
    sorted_indices = np.argsort(rel_paths)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('demo/'+clip_name+'.mp4', fourcc, 10.0, (w,h))
    # save_path = 'demo/'+clip_name+'.mp4'
    save_path = os.path.join('demo', model_dir, batch['dataset'][0], f'{clip_name}.mp4')
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving to:', save_path)

    # out = cv2.VideoWriter(save_path, fourcc, 10.0, (w, h))
    out = cv2.VideoWriter(save_path, fourcc, 8.0, (w, h))
    for idx in tqdm(sorted_indices):
        batch = rel_batches[idx]

        # LOAD gaze point
        gaze_pt_pred = batch['gp_pred']
        
        # LOAD LAH
        lah_pred = batch['lah_pred']
        # peform arg max for lah
        num_people = gaze_pt_pred.shape[1]
        pair_indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
        lah_pred_argmax = torch.zeros_like(lah_pred)
        for i in range(num_people):
            valid_indices = torch.where((pair_indices[:, 1]==i).int()*(pair_indices[:, 0]!=0).int())[0]
            if valid_indices.shape[0]>0:
                max_val, max_idx = torch.max(lah_pred[0][valid_indices], 0)
                lah_pred_argmax[0][valid_indices[max_idx]] = max_val
                
        # LOAD LAEO
        laeo_pred = batch['laeo_pred']
        # peform arg max for laeo
        laeo_pred_argmax = torch.zeros_like(laeo_pred)
        for i in range(num_people):
            valid_indices = torch.where((pair_indices[:, 1]==i).int()*(pair_indices[:, 0]!=0).int())[0]
            if valid_indices.shape[0]>0:
                max_val, max_idx = torch.max(laeo_pred[0][valid_indices], 0)
                laeo_pred_argmax[0][valid_indices[max_idx]] = max_val
                
        # LOAD SA
        coatt_pred = batch['coatt_pred']
        
        # concatenate social gaze predictions
        social_preds = [lah_pred_argmax, laeo_pred_argmax, coatt_pred]
        
        # get frame with visualizations
        frame = visualize_gaze(batch, social_preds)

        # get frame with coatt visualizations
        frame = visualize_coatt(frame, batch)

        # coatt_level_pred = batch['coatt_level_pred'][0]
        # coatt_hm_pred = batch['coatt_hm_pred'][0, :, :, :]

        # for token_idx in range(coatt_level_pred.shape[0]):
            # coatt_level_pred_token = coatt_level_pred[token_idx, :]
            # print(f'Token {token_idx}:', coatt_level_pred_token)
            # coatt_hm_pred_token = coatt_hm_pred[token_idx, :, :]
            # print(f'CoAtt HM Token {token_idx} sum:', torch.sum(coatt_hm_pred_token))
        
        # gaze_hm_pred = batch['hm_pred'][0, :, :, :]
        # for head_idx in range(len(gaze_hm_pred)):
        #     gaze_hm_pred_head = gaze_hm_pred[head_idx, :, :]
        #     head = batch['head_bboxes'][0][head_idx].clone()
        #     if torch.sum(head) == 0:
        #         continue

        #     # resize and normalize heatmap, then overlay on frame
        #     gaze_hm_pred_head = (gaze_hm_pred_head - torch.min(gaze_hm_pred_head.view(-1))) / (torch.max(gaze_hm_pred_head.view(-1)) - torch.min(gaze_hm_pred_head.view(-1)))
        #     gaze_hm_pred_head = cv2.resize(gaze_hm_pred_head.cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        #     gaze_hm_pred_head = (gaze_hm_pred_head * 255).astype(np.uint8)
        #     gaze_hm_pred_head = cv2.applyColorMap(gaze_hm_pred_head, cv2.COLORMAP_JET)
        #     frame = cv2.addWeighted(frame, 0.7, gaze_hm_pred_head, 0.3, 0)
        
        # write frame to output video
        out.write(frame)
    print('DONE!')
        
if __name__=='__main__':

    checkpoints_dir = "checkpoints"
    # model_dir = "combined_social/2025-10-02/09-40-42_COA_True_SOC_False"
    # model_dir = "combined_social/2025-10-02/09-40-26_COA_True_SOC_True"
    # model_dir = "combined_social/2025-10-02/09-40-32_COA_False_SOC_True"
    # model_dir = "combined_social/2025-10-03/09-21-05_COA_True_SOC_False"

    # model_dir = "combined_social/2025-10-08/12-53-07_COA_False_SOC_True"
    model_dir = "combined_social/2025-10-08/13-39-50_COA_True_SOC_False"

    dataset_name = 'videocoatt'
    # dataset_name = 'childplay'
    # dataset_name = 'videoattentiontarget'
    # dataset_name = 'uco_laeo'
    results_name = 'test-predictions.pickle'
    results_path = os.path.join(checkpoints_dir, model_dir, dataset_name, results_name)
    # print('Loading data...')
    # s_time = time.time()
    # with open(results_path, 'rb') as fp:
    #     results = pickle.load(fp)
    # print('Done! Time:', time.time()-s_time)

    print('Loading data...')
    s_time = time.time()
    with open(results_path, 'rb') as fp:
        results = joblib.load(fp)
    print('Done! Time:', time.time()-s_time)

    data_id_list = []
    for i in range(len(results)):
        result = results[i]
        mid_idx = len(result['path'])//2
        data_id_scene = result['path'][mid_idx][0].split('/')[-3]
        data_id_vid = result['path'][mid_idx][0].split('/')[-2]
        data_id = os.path.join(data_id_scene, data_id_vid)
        data_id_list.append(data_id)
    print('Unique data ids:', set(data_id_list))

    clip_name_list = []

    if dataset_name=='videocoatt':
        # clip_name_list.append('test/10')
        # clip_name_list.append('test/15')
        # clip_name_list.append('test/20')
        # clip_name_list.append('test/100')
        clip_name_list = set(data_id_list)
    elif dataset_name=='videoattentiontarget':
        # clip_name_list.append('CBS This Morning/11118_11359')
        # clip_name_list.append('CBS This Morning/2487_2818')
        # clip_name_list.append('CBS This Morning/12137_12258')
        clip_name_list = set(data_id_list)
    elif dataset_name=='uco_laeo':
        clip_name_list.append('frames/got05')
    elif dataset_name=='childplay':
        # clip_name_list.append('images/1Ab4vLMMAbY_412-554')
        # clip_name_list.append('images/1Ab4vLMMAbY_2354-2439')
        clip_name_list = set(data_id_list)

    # iterate each video
    for clip_name in clip_name_list:
        compute(results, clip_name)