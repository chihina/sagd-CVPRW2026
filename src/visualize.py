import pickle
import torch
from tqdm import tqdm
import itertools
import numpy as np
import os
import cv2
import sys

def get_root(batch):
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
    num_people = len(batch['head_bboxes'][0])
    for i in range(1, num_people):
        # draw head bbox
        head_bbox = batch['head_bboxes'][0][i].clone()
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
        frame = cv2.circle(frame, gp_pred.astype(np.int16), radius, color, thickness=-1)

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
    text_offset = 0
    for t, task in enumerate(['lah', 'laeo', 'coatt']):
        social_person = social_persons[t]
        text_offset += 30
        # iterate over heads
        for i in range(num_people):
            color = colors[i]
            # write task over head bbox
            flag = 0
            if task=='coatt':
                if social_person[i]!=[]:
                    flag = 1
            else:
                if social_person[i]!=-1:
                    flag = 1
            if flag:
                social_person[i] = str(set(social_person[i]))
                head_bbox = batch['head_bboxes'][0][i].clone()
                head_bbox = perc_to_pixel(head_bbox, frame_height, frame_width).int().cpu().numpy()
                org = (head_bbox[0], head_bbox[3] + text_offset)
                frame = cv2.putText(frame, task+': '+str(social_person[i]), org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    return frame

# main function
def compute(results, clip_name):
        
    print('Getting frames...')
    rel_batches = []; rel_paths = []
    for batch in tqdm(results):
        
        if batch['dataset'][0]=='videoattentiontarget':
            breakpoint()
        # filter based on clip name
        t = len(batch['path'][0])
        middle_frame_idx = int(t/2)
        path = batch['path'][0][middle_frame_idx]
        cname = '/'.join(path.split('/')[:-1])
        if cname!=clip_name:
            continue
            
        rel_batches.append(batch)
        rel_paths.append(path)
    
    # read one frame
    root = get_root(batch)
    frame_tmp = cv2.imread(os.path.join(root, path))
    h, w = frame_tmp.shape[:2]
    
    # iterate over sorted frames
    print('Getting results...')
    sorted_indices = np.argsort(rel_paths)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('demo/'+clip_name+'.mp4', fourcc, 10.0, (w,h))
    save_path = 'demo/'+clip_name+'.mp4'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving to:', save_path)
    out = cv2.VideoWriter(save_path, fourcc, 10.0, (w, h))
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
        
        # write frame to output video
        out.write(frame)
    print('DONE!')
        
if __name__=='__main__':
    
    results_path = 'test-predictions.pickle'
    print('Loading data...')
    with open(results_path, 'rb') as fp:
        results = pickle.load(fp)

    for i in range(len(results)):
        result = results[i]
        # print(result['path'][0])
        coatt_level_pred = result['coatt_level_pred'][0]
        print('coatt_level_pred shape:', coatt_level_pred)
        # coatt_hm_pred = result['coatt_hm_pred']
        # coatt_hm_pred = coatt_hm_pred.view(-1)

    clip_name = 'test/10'
    compute(results, clip_name)