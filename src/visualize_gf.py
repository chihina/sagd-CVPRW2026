import pickle
import torch
from tqdm import tqdm
import itertools
import numpy as np
import os
import cv2
import sys

def get_root(batch):
    root_gf = '/home-local/nakatani/dataset/gazefollow_extended'
    root_coatt = '/home-local/nakatani/dataset/VideoCoAtt_Dataset'
    root_vat = '/idiap/resource/database/VideoAttentionTarget'
    root_laeo = '/idiap/home/nchutisilp/UCO-LAEO/'
    root_childplay = '/idiap/temp/stafasca/data/ChildPlay'
    root = ''
    if 'dataset' in batch.keys():
        if batch['dataset'][0]=='videoattentiontarget':
            root = os.path.join(root_vat, 'images')
        elif batch['dataset'][0]=='childplay':
            root = root_childplay
        elif batch['dataset'][0]=='laeo':
            root = os.path.join(root_laeo, 'images_Idiap')
        elif batch['dataset'][0]=='coatt':
            root = os.path.join(root_coatt, 'images_nk')
        elif batch['dataset'][0]=='gazefollow':
            root = os.path.join(root_gf)

    return root
            
# auxiliary functions for drawing
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), 
           (0, 255, 255), (0, 255, 165), (255, 165, 0), (0, 165, 255), (255, 0, 165), (165, 255, 0), 
           (165, 255, 165), (255, 165, 165), (165, 165, 255), (165, 165, 0), (165, 85, 85), (85, 0, 85), (85, 85, 85)]

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
    path_list = batch['path'][0]
    # frame = cv2.imread(os.path.join(root, batch['path'][0]))
    frame = cv2.imread(os.path.join(root, path_list[len(path_list)//2]))
    frame_height, frame_width = frame.shape[:2]

    # iterate over all people
    # print('batch keys:', batch.keys())
    num_people = len(batch['head_bboxes'][0])
    for i in range(1, num_people):

        # check if head bbox is valid
        head_bbox = batch['head_bboxes'][0][i].clone()
        if torch.sum(head_bbox) == 0:
            continue

        # draw head bbox
        head_bbox = perc_to_pixel(head_bbox, frame_height, frame_width).int().cpu().numpy()
        color = colors[i]
        thickness = 3
        frame = cv2.rectangle(frame, head_bbox[:2], head_bbox[2:], color, thickness)

        # write pid
        org = (head_bbox[0], head_bbox[1] + 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color_text = (255, 255, 255)
        frame = cv2.putText(frame, str(i), org, font, fontScale, color_text, thickness, cv2.LINE_AA)

        # check if head inout prediction is over threshold
        # head_inout_pred = batch['inout_pred'][0][i]
        head_inout_pred = batch['inout_pred'][i-1]
        # head_inout_gt = batch['inout_gt'][0][i]
        head_inout_gt = batch['inout_gt'][i-1]
        head_inout_thresh = 0.2
        if head_inout_pred < head_inout_thresh:
            continue

        # draw gaze vector
        # gaze_vec_pred = batch['gv_pred'][0][i]
        gaze_vec_pred = batch['gv_pred'][i-1]
        start = (head_bbox[:2] + head_bbox[2:])/2
        # gv_pred = gaze_vec_pred[0][i].clone().cpu().numpy()
        gv_pred = gaze_vec_pred.clone().cpu().numpy()
        end = start + gv_pred*100
        # frame = cv2.line(frame, start.astype(np.int16), end.astype(np.int16), color_text, thickness)

        # draw gaze point
        # gaze_pt_pred = batch['gp_pred'][0][i]
        gaze_pt_pred = batch['gp_pred'][i-1]
        start = (head_bbox[:2] + head_bbox[2:])/2
        # gp_pred = gaze_pt_pred[0][i].clone().cpu().numpy()
        gp_pred = gaze_pt_pred.clone().cpu().numpy()
        gp_pred *= [frame_width, frame_height]
        frame = cv2.line(frame, start.astype(np.int16), gp_pred.astype(np.int16), color, thickness)
        radius = 10
        frame = cv2.circle(frame, gp_pred.astype(np.int16), radius, color, thickness=-1)

        # drwa gaze point (GT)
        gaze_pt_gt = batch['gp_gt'][0][i]
        # gaze_pt_gt = batch['gp_gt'][i]
        gp_gt = gaze_pt_gt.clone().cpu().numpy()
        gp_gt *= [frame_width, frame_height]
        frame = cv2.line(frame, start.astype(np.int16), gp_gt.astype(np.int16), (0, 0, 0), thickness)
        radius = 10
        frame = cv2.circle(frame, gp_gt.astype(np.int16), radius, (0, 0, 0), thickness=-1)

    # iterate over tasks
    indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
    lah_person = np.zeros(num_people, dtype=np.int16)-1
    laeo_person = np.zeros(num_people, dtype=np.int16)-1
    coatt_person = []
    for i in range(num_people):
        coatt_person.append([])
    social_persons = [lah_person, laeo_person, coatt_person]
    
    thresholds = [0.6, 0.6, 0.4]
    for t, task in enumerate(['lah', 'laeo', 'coatt']):
        social_pred = social_preds[t]
        social_person = social_persons[t]
        thres = thresholds[t]
        # iterate over pairs
        for pair_num, pair_score in enumerate(social_pred[0]):
            if pair_score > thres:
                pair_indices = indices[pair_num]
                if pair_indices[0]==0 or pair_indices[1]==0:
                    continue
                if task=='coatt':
                    social_person[pair_indices[0]].append(pair_indices[1].item())
                    social_person[pair_indices[1]].append(pair_indices[0].item())
                else:
                    if task=='laeo':
                        social_person[pair_indices[0]] = pair_indices[1].item()
                    social_person[pair_indices[1]] = pair_indices[0].item()

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
    # coatt_level_pred = batch['coatt_level_pred'][0]
    # print('coatt_level_pred shape:', coatt_level_pred.shape)
    # print('coatt_level_pred shape:', coatt_level_pred[:, :])
    coatt_hm_pred = batch['coatt_hm_pred'][0, 0, :, :]
    coatt_hm_pred = (coatt_hm_pred - torch.min(coatt_hm_pred.view(-1))) / (torch.max(coatt_hm_pred.view(-1)) - torch.min(coatt_hm_pred.view(-1)))
    coatt_hm_pred = cv2.resize(coatt_hm_pred.cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
    coatt_hm_pred = (coatt_hm_pred * 255).astype(np.uint8)
    coatt_hm_pred = cv2.applyColorMap(coatt_hm_pred, cv2.COLORMAP_JET)
    frame = cv2.addWeighted(frame, 0.7, coatt_hm_pred, 0.3, 0)

    return frame
    
# main function
def compute(results, clip_name_list):
        
    print('Getting frames...')
    rel_batches = []; rel_paths = []
    for batch in tqdm(results):
        
        if batch['dataset'][0]=='videoattentiontarget':
            breakpoint()
        # filter based on clip name
        t = len(batch['path'][0])
        middle_frame_idx = int(t/2)
        path = batch['path'][0][middle_frame_idx]

        # if path!=clip_name:
            # continue

        if not path in clip_name_list:
            continue

        rel_batches.append(batch)
        rel_paths.append(path)
    
    # read one frame
    # root = get_root(batch)
    # frame_tmp = cv2.imread(os.path.join(root, path))
    # h, w = frame_tmp.shape[:2]
    
    # iterate over sorted frames
    print('Getting results...')
    sorted_indices = np.argsort(rel_paths)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('demo/'+clip_name+'.mp4', fourcc, 10.0, (w,h))
    # save_path = 'demo/'+clip_name+'.mp4'
    # save_dir = os.path.dirname(save_path)
    save_dir = os.path.join('demo', 'gazefollow')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving to:', save_dir)
    # out = cv2.VideoWriter(save_path, fourcc, 10.0, (w, h))
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
        # frame = visualize_coatt(frame, batch)
        
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
        # out.write(frame)

        # save frame

        data_id = batch['path'][0][0]
        frame_save_path = os.path.join(save_dir, data_id)
        frame_save_dir = os.path.dirname(frame_save_path)
        if not os.path.exists(frame_save_dir):
            os.makedirs(frame_save_dir)
        cv2.imwrite(frame_save_path, frame)

    print('DONE!')
        
if __name__=='__main__':
    
    results_path = 'test-predictions.pickle'
    print('Loading data...')
    with open(results_path, 'rb') as fp:
        results = pickle.load(fp)

    # for i in range(len(results)):
        # result = results[i]
        # print(result['path'][0])
        # coatt_level_pred = result['coatt_level_pred'][0]
        # print('coatt_level_pred shape:', coatt_level_pred)
        # coatt_hm_pred = result['coatt_hm_pred']
        # coatt_hm_pred = coatt_hm_pred.view(-1)

    # videocoatt
    # clip_name = 'test/10'
    # clip_name = 'test/100'

    # gazefollow
    clip_name_list = []
    clip_name_list.append('test2/00000000/00000001.jpg')
    clip_name_list.append('test2/00000000/00000010.jpg')
    clip_name_list.append('test2/00000000/00000020.jpg')
    clip_name_list.append('test2/00000000/00000030.jpg')
    clip_name_list.append('test2/00000000/00000040.jpg')
    clip_name_list.append('test2/00000000/00000050.jpg')

    compute(results, clip_name_list)