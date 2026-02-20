import os
import pandas as pd
import cv2
import numpy as np
import glob
from tqdm import tqdm

# dataset_type = 'childplay'
dataset_type = 'vat'

if dataset_type == 'vat':
    dataset_dir = os.path.join('data', 'videoattentiontarget')
    data_id = '1738_1919'
    data_id_list = [data_id]
elif dataset_type == 'childplay':
    dataset_dir = os.path.join('data', 'ChildPlay-gaze')
    csv_file_list = os.listdir(os.path.join(dataset_dir, 'annotations', 'test'))
    data_id_list = [f.replace('.csv', '') for f in csv_file_list if f.endswith('.csv')]
    data_id_list = sorted(data_id_list)
    # data_id = '31lG75MDwSA_4686-4764'
    # data_id = '1Ab4vLMMAbY_412-554'
    # data_id_list = [data_id]

# co_att_type = 'pp'  # 'pp' or 'sp'
co_att_type = 'sp'  # 'pp' or 'sp'

coatt_count = 0

for data_num, data_id in enumerate(data_id_list):
    annotation_path = f"data/VSGaze/{dataset_type}_test.h5"

    annotations = pd.read_hdf(annotation_path)
    annotations = annotations.groupby('path')
    paths = list(annotations.groups.keys())
    paths = np.array(paths)
    print(paths)

    if dataset_type == 'vat':
        data_dir = glob.glob(os.path.join(dataset_dir, 'images', 'CBS*', data_id))[0]
        image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')])
    elif dataset_type == 'childplay':
        image_paths = os.path.join(dataset_dir, 'images', data_id)
        image_paths = sorted([os.path.join(image_paths, f) for f in os.listdir(image_paths) if f.endswith('.jpg')])

    # generate a video with annotations
    video_output_path = os.path.join('visualizations_annotations', f'{dataset_type}', f'{data_id}_annotated_{co_att_type}.mp4')
    if not os.path.exists(os.path.dirname(video_output_path)):
        os.makedirs(os.path.dirname(video_output_path))

    dist_thr = 0.05  # threshold for co-attention (not used in this code)

    for frame_idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc=f'Processing {data_id}'):
        # Load image
        image = cv2.imread(image_path)
        image_h, image_w = image.shape[:2]

        frame_number = frame_idx + 1  # assuming frames are 1-indexed in annotations

        image_name = os.path.basename(image_path)
        if dataset_type == 'childplay':
            search_id = f'images/{data_id}/{image_name}'
        elif dataset_type == 'vat':
            search_id = f'CBS_This_Morning/{data_id}/{image_name}'

        if search_id not in paths:
            print(f'Annotation for {search_id} not found.')
            continue
        annotation = annotations.get_group(search_id)

        head_bboxes = annotation['head_bboxes'].values[0]
        gaze_points = annotation['gaze_points'].values[0]
        inout = annotation['inout'].values[0]
        pairs = annotation['pairs'].values[0]
        coatt_pairs = annotation['coatt_pairs'].values[0]

        # Draw annotations
        for head_idx in range(len(head_bboxes)):
            head = head_bboxes[head_idx]
            if np.sum(head == 0) == 4:
                continue

            x1, y1, x2, y2 = head
            x1, x2 = x1 * image_w, x2 * image_w
            y1, y2 = y1 * image_h, y2 * image_h
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if inout[head_idx] == 0:
                continue  # skip if gaze is out-of-frame

            gaze = gaze_points[head_idx]
            gaze_x, gaze_y = gaze
            # if gaze_x == -1 and gaze_y == -1:
                # cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 2)
                # continue  # skip if gaze point is invalid
                
            # visualize gaze arrow
            gaze_x, gaze_y = int(gaze[0] * image_w), int(gaze[1] * image_h)
            # cv2.arrowedLine(image, ((x1 + x2) // 2, (y1 + y2) // 2), (gaze_x, gaze_y), (255, 0, 0), 2, tipLength=0.03)
            
            # Default color for bounding box (green)
            bbox_color = (0, 255, 0)

            gaze_dist_all = gaze - gaze_points
            gaze_dist_all = np.linalg.norm(gaze_dist_all, axis=1)
            gaze_dist_all[head_idx] = np.inf  # ignore self-distance
            gaze_dist_all[inout == 0] = np.inf  # ignore out-of-frame gaze points

            # print(pairs)
            # print(coatt_pairs)

            if co_att_type == 'pp':
                for other_head_idx in range(len(head_bboxes)):
                    if other_head_idx == head_idx:
                        continue
                    if gaze_dist_all[other_head_idx] < dist_thr:
                        bbox_color = (0, 0, 255)  # change color to red if co-attention detected
                        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)
                        x1_other, y1_other, x2_other, y2_other = head_bboxes[other_head_idx]
                        x1_other, x2_other = x1_other * image_w, x2_other * image_w
                        y1_other, y2_other = y1_other * image_h, y2_other * image_h
                        x1_other, y1_other, x2_other, y2_other = map(int, [x1_other, y1_other, x2_other, y2_other])
                        cv2.rectangle(image, (x1_other, y1_other), (x2_other, y2_other), bbox_color, 2)
                        coatt_count += 1
            elif co_att_type == 'sp':
                for pair_idx, pair in enumerate(pairs):
                    coatt = coatt_pairs[pair_idx]
                    if coatt == 1:
                        head_idx, target_idx = pair
                        print(head_idx, target_idx)
                        head = head_bboxes[head_idx]
                        target = head_bboxes[target_idx]
                        if np.sum(head) == 0 or np.sum(target) == 0:
                            continue
                        hx1, hy1, hx2, hy2 = head
                        tx1, ty1, tx2, ty2 = target
                        hx1, hx2 = hx1 * image_w, hx2 * image_w
                        hy1, hy2 = hy1 * image_h, hy2 * image_h
                        tx1, tx2 = tx1 * image_w, tx2 * image_w
                        ty1, ty2 = ty1 * image_h, ty2 * image_h
                        hx1, hy1, hx2, hy2 = map(int, [hx1, hy1, hx2, hy2])
                        tx1, ty1, tx2, ty2 = map(int, [tx1, ty1, tx2, ty2])

                        cv2.rectangle(image, (hx1, hy1), (hx2, hy2), (0, 0, 0), 2)
                        cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (0, 0, 0), 2)
                        coatt_count += 1

        # Save annotated frame temporarily
        temp_frame_path = f'temp_frame_{frame_idx:05d}.jpg'
        cv2.imwrite(temp_frame_path, image)

        # Append frame to video
        if frame_idx == 0:
            height, width, layers = image.shape
            video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        video.write(image)
        os.remove(temp_frame_path)
    video.release()
    print(f'Video saved to {video_output_path}')

    if data_num >= 10:
        break  # process only first 3 videos for demo purposes

    print(f'Total co-attention instances: {coatt_count}')
print('Done.')