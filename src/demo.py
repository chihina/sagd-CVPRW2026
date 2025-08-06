
import os
import sys
import shlex
import shutil
import argparse
import importlib
import datetime as dt
from tqdm import tqdm
import subprocess as sp
from pathlib import Path
from omegaconf import OmegaConf
from termcolor import colored
import itertools
import pandas as pd

import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose

from boxmot import DeepOCSORT, BYTETracker, OCSORT, BoTSORT

# Yolo imports
sys.path.insert(0, "/idiap/temp/agupta/src/yolov5-crowdhuman/")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords

# MiDaS imports
sys.path.append("/idiap/temp/stafasca/repos/MiDaS")
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# GazeInteract imports
# EXP_PATH = "/idiap/temp/agupta/src/gaze_interact/experiments/2023-11-17/00-24-57" # GazeInteract, static, no spk
# EXP_PATH = "/remote/idiap.svm/user.active/agupta/gaze_interact/experiments/2025-02-04/16-54-36" # MTGS-static-DINO (GF 448 init)
EXP_PATH = "/remote/idiap.svm/temp.perception02/agupta/gaze_interact/experiments/2025-03-21/14-06-38" # MTGS-static-DINO-448
sys.path.insert(0, EXP_PATH)
from src.utils import spatial_argmax2d, square_bbox#, expand_bbox
#from src.utils.geometry import compute_gaze3d


# ================================ ARGS ================================ #
parser = argparse.ArgumentParser(description="Predict gaze on videos")
parser.add_argument("--input-dir", type=str, default="/idiap/temp/stafasca/data/ChildPlay/clips", 
                    help="Name of the folder where to find the input.")
parser.add_argument("--input-filename", type=str, help="Names of the clip file to process (with extension).")
parser.add_argument("--output-dir", type=str, default="demo", help="Name of the folder where to save the output.")
parser.add_argument("--head-tracks-file", type=str, help="Path to the head tracks file.")
parser.add_argument("--heatmap-pid", type=int, default=-1, help="pid of the person to draw the heatmap of.")

parser.add_argument("--alpha", type=float, default=0.5, help="Controls the weight of the heatmap w.r.t the image.")
parser.add_argument("--fs", type=float, default=1.2, help="The font scale to use when drawing text on the image.")
parser.add_argument("--thickness", type=int, default=12, help="Controls the thickness of lines and rectangles on the image.")
parser.add_argument("--gaze-pt-size", type=int, default=24, help="Controls the size of the gaze point on the image.")
parser.add_argument("--head-center-size", type=int, default=24, help="Controls the size of the head center point on the image.")

parser.add_argument("--show-exp-path", action='store_true', help="Whether to draw the checkpoint path.")
parser.add_argument('--no-show-exp-path', dest='show_exp_path', action='store_false')
parser.set_defaults(show_exp_path=False)

parser.add_argument("--show-frame-nb", action='store_true', help="Whether to draw the frame number.")
parser.add_argument('--no-show-frame-nb', dest='show_frame_nb', action='store_false')
parser.set_defaults(show_frame_nb=False)

parser.add_argument("--show-gaze-vec", action='store_true', help="Whether to draw the gaze vector.")
parser.add_argument('--no-show-gaze-vec', dest='show_gaze_vec', action='store_false')
parser.set_defaults(show_gaze_vec=False)

args = parser.parse_args()


# =============================== GLOBALS =============================== #
TERM_COLOR = "cyan"
MAX_DEPTH = 65535
DEPTH_MEAN = [0.43612]
DEPTH_STD = [0.30262]
DET_THR = 0.4 # head detection threshold
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]
COLOR_NAMES = ["mediumvioletred", "green", "dodgerblue", "crimson", "goldenrod", "DarkSlateGray", 
               "saddlebrown", "purple", "teal"]
COLORS = [(199, 21, 133), (0, 128, 0), (30, 144, 255), (220, 20, 60), (218, 165, 32), 
          (47, 79, 79), (139, 69, 19), (128, 0, 128), (0, 128, 128)]


#EXP_PATH = "/idiap/temp/stafasca/projects/rinnegan/experiments/2023-07-17/23-56-34" # depth
#EXP_PATH = "/idiap/temp/stafasca/projects/rinnegan/experiments/2023-07-31/00-27-50" # 3D gaze

# DEVICE='cpu'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(colored(f"Using device: {DEVICE}", TERM_COLOR))

# ========================= UTILITY FUNCTIONS =========================== #
def expand_bbox(bbox, img_w, img_h, k=0.1):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bbox[0] = max(0, bbox[0] - k * w)
    bbox[1] = max(0, bbox[1] - k * h)
    bbox[2] = min(img_w, bbox[2] + k * w)
    bbox[3] = min(img_h, bbox[3] + k * h)
    return bbox

def is_inside(head_bbox, gaze_pt):
    if gaze_pt[0]>head_bbox[0] and gaze_pt[0]<head_bbox[2] and gaze_pt[1]>head_bbox[1] and gaze_pt[1]<head_bbox[3]:
        return True
    return False

def detect_heads(image, model, img_size=640, conf_thres=0.25, iou_thres=0.45,  device="cpu"):

    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    # Pre-process image
    image_input = letterbox(image, img_size, stride=stride)[0]
    image_input = image_input.transpose(2, 0, 1)  # to 3x416x416
    image_input = torch.from_numpy(image_input).unsqueeze(0).float().to(device)
    image_input = image_input / 255.0 if image_input.max() > 1 else image_input

    # Inference
    with torch.no_grad():
        predictions = model(image_input)[0]

    # Apply NMS
    predictions = non_max_suppression(predictions, conf_thres, iou_thres)[0]
    predictions[:, :4] = scale_coords(image_input.shape[2:], predictions[:, :4], image.shape).round()
    predictions = predictions.cpu().numpy()

    # Separate head from person detections
    class_names = model.names
    h_mask = (predictions[:, -1] == class_names.index('head'))
    h_detections = predictions[h_mask, :-1]

    p_mask = (predictions[:, -1] == class_names.index('person'))
    p_detections = predictions[p_mask, :-1]
    
    return h_detections

# show_exp_path, show_frame_nb, show_gaze_vec, alpha, thickness, fs, gaze_pt_size, head_center_size
# thickness: head_bbox rectangle: int(1.5 * thickness), gaze line: int(1.5 * thickness), gaze vec line: int(1.5 * thickness)
# fs: hm_pid_text: fs, bbox header: fs, frame_nb: fs, exp_path: fs


def draw_gaze(
    image: np.ndarray,
    social_preds,
    head_bboxes,
    gaze_points,
    gaze_vecs,
    inouts,
    pids,
    gaze_heatmaps,
    heatmap_pid = None,
    frame_nb = None,
    exp_path = None,
    colors = COLORS,
    alpha: float = 0.5,
    io_thr: float = 0.7, 
    gaze_pt_size: int = 10,
    head_center_size: int = 10,
    thickness: int = 4,
    fs: float = 0.6,
):
    """
    Function to draw gaze results for a single person.

    Args:
        image: input image to draw gaze for.
        gaze_points: 2d coordinates of the gaze target points.
        inouts: vector of scores between 0 and 1.
        head_bboxes: head bounding boxes.
        person_id: id of the person for which to draw gaze heatmap when given.
        gaze_heatmaps: the predicted gaze heatmaps.

    Returns:
        canvas or canvas_ext: the output image with gaze predictions drawn, and optionally, the other modalities.
    """
    
    # Create canvas on which to draw predictions
    img_h, img_w, img_c = image.shape
    canvas = image.copy()
    
    # Scale of the drawing according to image resolution
    scale = max(img_h, img_w) / 1920
    fs *= scale
    thickness = int(scale * thickness)
    gaze_pt_size = int(scale * gaze_pt_size)
    head_center_size = int(scale * head_center_size)
    
    # Draw heatmap
    if heatmap_pid is not None:
        if len(gaze_heatmaps) == 0:
            raise ValueError("gaze_heatmaps must be provided if heatmap_pid is provided.")
        mask = (pids == heatmap_pid)
        if mask.sum() == 1: # only if detection found
            gaze_heatmap = gaze_heatmaps[mask]
            heatmap = TF.resize(gaze_heatmap, (img_h, img_w), antialias=True).squeeze().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap = cm.inferno(heatmap) * 255 
            canvas = ((1 - alpha) * image + alpha * heatmap[..., :3]).astype(np.uint8)

            # Write pid being used for the heatmap
            hm_pid_text = f"Heatmap PID: {heatmap_pid}"
            (w_text, h_text), _ = cv2.getTextSize(hm_pid_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ul = (img_w - w_text - 20, img_h - h_text - 15)
            br = (img_w, img_h)
            cv2.rectangle(canvas, ul, br, (0, 0, 0), -1)
            hm_pid_text_loc = (img_w - w_text - 10, img_h - 10)
            cv2.putText(canvas, hm_pid_text, hm_pid_text_loc, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)   

    # Draw head bboxes  
    if len(head_bboxes) > 0:
        if len(pids) == 0:
            raise ValueError("pids must be provided if head_bboxes is provided.")
            
        # Convert to numpy
        head_bboxes = head_bboxes.numpy() if isinstance(head_bboxes, torch.Tensor) else np.array(head_bboxes)
        if head_bboxes.max() <= 1.1: # TODO: check why values can be slightly > 1. when using square_bbox
            head_bboxes = head_bboxes * np.array([img_w, img_h, img_w, img_h])
        head_bboxes = head_bboxes.astype(int)
        
        # Compute head center
        head_centers = np.hstack([(head_bboxes[:,[0]] + head_bboxes[:,[2]]) / 2, (head_bboxes[:,[1]] + head_bboxes[:,[3]]) / 2])
        head_centers = head_centers.astype(int)
        
        gaze_available = (len(gaze_points) > 0)
        if gaze_available and (len(inouts) == 0):
            raise ValueError("inouts must be provided if gaze_pts is provided.")
            
        if gaze_available:
            gaze_points = gaze_points.numpy() if isinstance(gaze_points, torch.Tensor) else np.array(gaze_points)
            if (gaze_points.max() <= 1.):
                gaze_points = gaze_points * np.array([img_w, img_h])
            gaze_points = gaze_points.astype(int)
            
        if gaze_vecs is not None:
            gaze_vecs = gaze_vecs.numpy() if isinstance(gaze_vecs, torch.Tensor) else np.array(gaze_vecs)
        
        for i, head_bbox in enumerate(head_bboxes):
            xmin, ymin, xmax, ymax = head_bbox
            head_radius = max(xmax-xmin, ymax-ymin) // 2
            pid = pids[i]
            color = colors[pid % len(colors)]
                            
            # Compute Head Center
            head_center = head_centers[i]
        
            head_bbox_ul = (xmin, ymin)
            head_bbox_br = (xmax, ymax)
            #cv2.rectangle(canvas, head_bbox_ul, head_bbox_br, color, thickness) # head bbox
            head_center_ul = head_center - (head_center_size // 2)
            head_center_br = head_center + (head_center_size // 2)
#             cv2.rectangle(canvas, head_center_ul, head_center_br, color, -1) # head center point
            cv2.circle(canvas, head_center, head_radius, color, thickness) # head circle
            
            # Draw header
            io = inouts[i] if inouts is not None else "-"
            header_text = f"Person {pid}"
            (w_text, h_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            
            header_ul =  (int(head_center[0] - w_text / 2), int(ymin - thickness / 2))
            header_br = (int(head_center[0] + w_text / 2), int(ymin + h_text + 5))
            cv2.rectangle(canvas, header_ul, header_br, color, -1) # header bbox
            cv2.putText(canvas, header_text, (header_ul[0], int(ymin + h_text)), cv2.FONT_HERSHEY_SIMPLEX, 
                        fs, (255, 255, 255), 1, cv2.LINE_AA) # header text
            
            #header_ul =  (xmin, int(ymin - thickness / 2))
            #header_br = (int(xmin + w_text), int(ymin + h_text + 5))
            #cv2.rectangle(canvas, header_ul, header_br, color, -1) # header bbox
            #cv2.putText(canvas, header_text, (xmin, int(ymin + h_text)), cv2.FONT_HERSHEY_SIMPLEX, 
            #            fs, (255, 255, 255), 1, cv2.LINE_AA) # header text
            
            if gaze_available and (io > io_thr):
                gp = gaze_points[i]
                vec = (gp - head_center)
                vec = vec / (np.linalg.norm(vec) + 0.000001)
                intersection = head_center + (vec * head_radius).astype(int)
                #cv2.line(canvas, head_center, gp, color, int(0.5 * thickness)) # UNCOMMENT
                cv2.line(canvas, intersection, gp, color, thickness)
                
                cv2.circle(canvas, gp, gaze_pt_size, color, -1)
                
            if gaze_vecs is not None:
                gv = gaze_vecs[i]
                cv2.arrowedLine(canvas, head_center, (head_center + 100 * gv).astype(int), color, thickness)
                
                
    # Write frame number
    if frame_nb is not None:
        frame_nb = str(frame_nb)
        (w_text, h_text), _ = cv2.getTextSize(frame_nb, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        nb_ul = (int((img_w - w_text) / 2), (img_h - h_text - 15))
        nb_br = (int((img_w + w_text) / 2), img_h)
        cv2.rectangle(canvas, nb_ul, nb_br, (0, 0, 0), -1)
        nb_text_loc = (int((img_w - w_text) / 2), (img_h - 10))
        cv2.putText(canvas, frame_nb, nb_text_loc, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA) 
        
    # Write experiment name
    if exp_path is not None:
        exp_path = "/".join(exp_path.split("/")[-4:])
        exp_text = f"Experiment: {exp_path}"
        (w_text, h_text), _ = cv2.getTextSize(exp_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ul = (0, img_h - h_text - 15)
        br = (w_text + 20, img_h)
        #cv2.rectangle(canvas, ul, br, (0, 0, 0), -1) # UNCOMMENT
        exp_text_loc = (10, img_h - 10)
        #cv2.putText(canvas, exp_text, exp_text_loc, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA) # UNCOMMENT
        
    # ============ draw social gaze ============= #
    num_people = len(head_bboxes)
    # iterate over tasks
    indices = torch.tensor(list(itertools.permutations(torch.arange(num_people+1), 2)))
    lah_person = np.zeros(num_people+1, dtype=np.int16)-1
    laeo_person = np.zeros(num_people+1, dtype=np.int16)-1
    coatt_person = []
    for i in range(num_people+1):
        coatt_person.append([])
    social_persons = [lah_person, laeo_person, coatt_person]
    
    thresholds = [0.7, 0.7, 0.4]
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 1.7*(img_h/720)
    thickness_text = int(5*(img_h/720))
    # iterate over heads
    for i in range(num_people):
        text_offset = 0
        pid = pids[i]
        color = colors[pid % len(colors)]
        io = inouts[i]
        if not io > io_thr:
            continue
        for t, task in enumerate(['lah', 'laeo', 'coatt']):
            social_person = social_persons[t]
            # write task over head bbox
            flag = 0
            if task=='coatt':
                if social_person[i+1]!=[]:
                    flag = 1
            else:
                if social_person[i+1]!=-1:
                    flag = 1
            if flag:
                head_bbox = head_bboxes[i]
                if task=='coatt':
                    text_offset += 46
                    org = (head_bbox[0]+25, head_bbox[3] + text_offset)
                    out_name = ",".join([ str(pids[z-1].item()) for z in list(set(social_person[i+1]))])
                    cv2.putText(canvas, 'SA '+out_name, org, font, fs, color, thickness_text, cv2.LINE_AA)
                else:
                    text_offset += 46
                    org = (head_bbox[0]+25, head_bbox[3] + text_offset)
                    out_text = task.upper()+' '+str(pids[social_person[i+1]-1].item())
                    cv2.putText(canvas, out_text, org, font, fs, color, thickness_text, cv2.LINE_AA)
        
                
    return canvas


def save_predictions(predictions, output_file, img_w, img_h):
    columns = ["frame_nb", "gaze_pt_x", "gaze_pt_y", "gaze_vec_x", "gaze_vec_y", "inout", 'lah_id', 'laeo_id', 'coatt_id', "pid", "xmin", "ymin", "xmax", "ymax"]
    df = pd.DataFrame(columns=columns)

    for prediction in predictions:
        frame_nb = prediction["frame_nb"]
        num_people = len(prediction["gaze_points"])
        pair_indices = torch.tensor(list(itertools.permutations(torch.arange(num_people+1), 2)))
        for k in range(num_people):
            gp = prediction["gaze_points"][k].numpy()
            gp_x, gp_y = gp
            gv = prediction["gaze_vecs"][k].numpy()
            gv_x, gv_y = gv
            io = prediction["inouts"][k].numpy().item()
            pid = prediction["pids"][k].item()
            head_bbox = prediction["head_bboxes"][k].numpy()
            xmin, ymin, xmax, ymax = head_bbox
            xmin, xmax = xmin / img_w, xmax / img_w
            ymin, ymax = ymin / img_h, ymax / img_h
            # social preds
            valid_indices = torch.where((pair_indices[:, 1]==(k+1)).int()*(pair_indices[:, 0]!=0).int())[0]
            lah = prediction['lah'][0][valid_indices]
            laeo = prediction['laeo'][0][valid_indices]
            coatt = prediction['coatt'][0][valid_indices]
            # save social preds as a dict
            social_gaze_pids = pair_indices[valid_indices][:, 0].numpy()
            lah_dict = {}; laeo_dict = {}; coatt_dict = {}
            for si, spid in enumerate(social_gaze_pids):
                lah_dict[spid] = lah[si].item()
                laeo_dict[spid] = laeo[si].item()
                coatt_dict[spid] = coatt[si].item()

            row = {"frame_nb": frame_nb, "gaze_pt_x": gp_x, "gaze_pt_y": gp_y, "gaze_vec_x": gv_x, "gaze_vec_y": gv_y, 
                   "inout": io, 'lah_id': lah_dict, 'laeo_id': laeo_dict, 'coatt_id': coatt_dict, "pid": pid, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            df.loc[len(df)] = row

    df.to_csv(output_file, index=False)
    

# ========================= GAZE PREDICTOR =========================== #
class GazePredictor:
    def __init__(self, exp_path, device="cpu"):
        self.exp_path = exp_path
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Unzip code files
        self.unzip_code_files()
        
        # Load config
        self.cfg = OmegaConf.load(os.path.join(self.exp_path, "src/conf/config.yaml"))
        
        # Initialize models
        self.gaze_predictor = self.init_gaze_predictor(self.cfg) 
        self.head_detector = self.init_head_detector()
        self.depth_extractor, self.depth_transform = self.init_depth_extractor()
        self.tracker = self.init_tracker()
    
    
    def unzip_code_files(self):
        # Unzip code files
        src_dir_path = os.path.join(self.exp_path, "src")
        if not os.path.exists(src_dir_path):
            src_zip_path = os.path.join(self.exp_path, "src.zip")
            if os.path.exists(src_zip_path):
                print(colored(f"Unpacking {src_zip_path}", TERM_COLOR))
                shutil.unpack_archive(src_zip_path, src_dir_path)
            else:
                raise FileNotFoundError(f"Couldn't find the code files in the experiment folder: {self.exp_path}")
    

    def init_depth_extractor(self):
        ckpt_path = "/idiap/temp/stafasca/weights/depth/dpt_large-midas-2f21e586.pt"
        depth_extractor = DPTDepthModel(path=ckpt_path, backbone="vitl16_384", non_negative=True)
        depth_extractor.to(self.device)
        depth_extractor.eval()
        
        # Transform
        depth_transform = Compose(
            [
                Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
            ]
        )
        
        return depth_extractor, depth_transform
    
    
    def init_head_detector(self):
        # Load head detector
        ckpt_path = "/idiap/temp/agupta/src/yolov5-crowdhuman/weights/crowdhuman_yolov5m.pt"
        head_detector = attempt_load(ckpt_path, map_location="cpu")
        head_detector.to(self.device)
        head_detector.eval()
        return head_detector
    
    
    def init_tracker(self):
        #tracker = BoTSORT(
        #    model_weights=Path('/idiap/temp/stafasca/weights/tracking/osnet_x0_25_msmt17.pt'),
        #    device=self.device,
        #    fp16=True,
        #)
        #tracker = DeepOCSORT(
        #  model_weights=Path('/idiap/temp/stafasca/weights/tracking/osnet_x0_25_msmt17.pt'),  # which ReID model to use
        #  device=self.device,  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
        #  fp16=True,  # wether to run the ReID model with half precision or not
        #)
        #tracker = BYTETracker()
        tracker = OCSORT(det_thresh=0.2, asso_threshold=0.1, max_age=300, inertia=0.5)
        return tracker
        
    
    def init_gaze_predictor(self, cfg):
        
        # Initialize Rinnegan architecture
        module_name = os.path.join(self.exp_path, "src/networks/interact_net_temporal.py") ### ===
        #module_name = os.path.join(self.exp_path, "src/rinnegan_multimae.py")
        
        spec = importlib.util.spec_from_file_location("rinnegan", module_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        gaze_predictor = module.InteractNet(image_size=cfg.data.image_size,\
                                            patch_size=cfg.model.sharingan.patch_size,\
                                            decoder_feature_dim=128,\
                                            decoder_use_bn=True,\
                                            temporal_context=cfg.data.temporal_context)
        
        # Load checkpoint
        ckpt_path = os.path.join(self.exp_path, "checkpoints/best.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Couldn't find the checkpoint in the experiment folder: {self.exp_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        checkpoint = {name.replace("model.", ""): value for name, value in checkpoint["state_dict"].items()}
        gaze_predictor.load_state_dict(checkpoint, strict=True)
        gaze_predictor.to(self.device)
        gaze_predictor.eval()

        # Cleanup
        del checkpoint

        return gaze_predictor
    
    
    def predict(self, image, head_bboxes=None, pids=None, depth=None):
        
        image_np = np.array(image)
        img_h, img_w, img_c = image_np.shape
        if head_bboxes is None:
            # ============== 1. Detect people =============== #
            raw_detections = detect_heads(image_np, self.head_detector,  device=self.device)
            detections = []
            for k, raw_detection in enumerate(raw_detections):
                bbox, conf = raw_detection[:4], raw_detection[4]
                if conf > DET_THR:
                    bbox = expand_bbox(bbox, img_w, img_h, k=0.1)
                    cls_ = np.array([0.])
                    detection = np.concatenate([bbox, conf[None], cls_])
                    detections.append(detection)
            detections = np.stack(detections) if len(detections) > 0 else np.empty((0, 6))     

            # ================ 2. Track people =============== #
            tracks = self.tracker.update(detections, image_np)
            if len(tracks) == 0: # sometimes tracker.update returns [] even when detections is not []
                return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
            pids = (tracks[:, 4] - 1).astype(int)
            head_bboxes = torch.from_numpy(tracks[:, :4]).float()
        
        t_head_bboxes = square_bbox(head_bboxes, img_w, img_h)
        num_people = len(head_bboxes)
        
        # =============== X. Extract depth =============== #
        #img_input = self.depth_transform({"image": image_np / 255.0})["image"]
        #with torch.no_grad():
        #    img_input = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
        #    depth = self.depth_extractor.forward(img_input)
        
        # Process and transform
        #max_depth = depth.max()
        #min_depth = depth.min()
        #depth = (depth - min_depth) / (max_depth - min_depth) # to [0, 1]  
        #image_size = self.cfg.model.rinnegan.image_size
        #depth = TF.resize(depth, (image_size, image_size), antialias=True)
        #depth = TF.normalize(depth, mean=DEPTH_MEAN, std=DEPTH_STD)
        #depth = depth.unsqueeze(0)

        # ========== 3. Extract and transform heads ========== #
        heads = []
        for bbox in t_head_bboxes:
            head = TF.resize(TF.to_tensor(image.crop(bbox.numpy())), (224, 224), antialias=True)
            heads.append(head)

        heads = torch.stack(heads)
        heads = TF.normalize(heads, mean=IMG_MEAN, std=IMG_STD)

        # ================= 4. Transform Image ================ #
        image_size = self.cfg.model.sharingan.image_size
        image = TF.to_tensor(image)
        image = TF.resize(image, (image_size, image_size), antialias=True)
        image = TF.normalize(image, mean=IMG_MEAN, std=IMG_STD)
        image = image.unsqueeze(0)

        #if depth is not None:
        #    depth = TF.to_tensor(depth) / MAX_DEPTH
        #    depth = TF.resize(depth, (image_size, image_size), antialias=True)
        #    depth = TF.normalize(depth, mean=DEPTH_MEAN, std=DEPTH_STD)
        #    depth = depth.unsqueeze(0)
        
        # ============== 5. Normalize head bboxes ============= #
        t_head_bboxes[:, 0] /= img_w
        t_head_bboxes[:, 1] /= img_h
        t_head_bboxes[:, 2] /= img_w
        t_head_bboxes[:, 3] /= img_h
        num_valid_people = len(t_head_bboxes)

        # =============== 6. build input sample =============== #
        sample = {}
        sample["image"] = image.unsqueeze(0).to(self.device)
        sample["depth"] = depth.repeat_interleave(num_people, dim=0).to(self.device) if depth is not None else depth
        sample['num_valid_people'] = num_valid_people
        # pad with extra person
        heads = torch.cat([torch.zeros((1, 3, 224, 224), dtype=torch.float32), heads]).to(self.device)
        t_head_bboxes = torch.cat([torch.zeros((1, 4), dtype=torch.float32), t_head_bboxes]).to(self.device)
        sample["heads"] = heads.unsqueeze(0).unsqueeze(0).to(self.device)
        sample["head_bboxes"] = t_head_bboxes.unsqueeze(0).unsqueeze(0).to(self.device)
        
        #list_pcd_vecs = []
        #focal = 1338.6479
        #gaze_point = torch.tensor([.1, .1], device=self.device)
        #for hbbox in t_head_bboxes:
        #    head_center = torch.stack([hbbox[[0, 2]].mean(), hbbox[[1, 3]].mean()]).to(self.device)
        #    gaze_vec_3d, pcd_vecs = compute_gaze3d(depth.squeeze(), head_center, gaze_point, focal, True)
        #    list_pcd_vecs.append(pcd_vecs)
        #list_pcd_vecs = torch.stack(list_pcd_vecs)
        #sample["pcd_vecs"] = list_pcd_vecs.to(self.device) # (n, h, w, 3)
        
        # ================== 7. predict gaze ================== #
        with torch.no_grad():
            _, gaze_vecs, gaze_heatmaps, inouts, lah, laeo, coatt = self.gaze_predictor(sample)
            gaze_heatmaps = gaze_heatmaps.squeeze(0).squeeze(0)[1:]
            gaze_vecs = gaze_vecs.squeeze(0).squeeze(0)[1:]
            gaze_points = spatial_argmax2d(gaze_heatmaps, normalize=True)
            lah = lah.squeeze(0)
            laeo = laeo.squeeze(0)
            coatt = coatt.squeeze(0)
            inouts = inouts.squeeze(0).squeeze(0)[1:]
            
        return gaze_heatmaps, gaze_points, gaze_vecs, inouts, lah, laeo, coatt, head_bboxes, pids


def get_social_preds(lah_pred, laeo_pred, coatt_pred, num_people):
    # peform arg max for lah
    pair_indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
    lah_pred_argmax = torch.zeros_like(lah_pred)
    for i in range(num_people):
        valid_indices = torch.where((pair_indices[:, 1]==i).int()*(pair_indices[:, 0]!=0).int())[0]
        if valid_indices.shape[0]>0:
            max_val, max_idx = torch.max(lah_pred[0][valid_indices], 0)
            lah_pred_argmax[0][valid_indices[max_idx]] = max_val

    # peform arg max for laeo
    laeo_pred_argmax = torch.zeros_like(laeo_pred)
    for i in range(num_people):
        valid_indices = torch.where((pair_indices[:, 1]==i).int()*(pair_indices[:, 0]!=0).int())[0]
        if valid_indices.shape[0]>0:
            max_val, max_idx = torch.max(laeo_pred[0][valid_indices], 0)
            laeo_pred_argmax[0][valid_indices[max_idx]] = max_val

    # concatenate social gaze predictions
    social_preds = [lah_pred_argmax, laeo_pred_argmax, coatt_pred]
    
    return social_preds


def main():

    start = dt.datetime.now()
    
    # Path magic
    video_file = os.path.join(args.input_dir, args.input_filename)
    basename, ext = os.path.splitext(args.input_filename)
    exp = "_".join(EXP_PATH.split("/")[-2:]) # [day, time]
    if args.heatmap_pid >= 0:
        output_file = os.path.join(args.output_dir, f"{basename}-pid{args.heatmap_pid}-exp{exp}-pred{ext}")
    else:
        output_file = os.path.join(args.output_dir, f"{basename}-exp{exp}-pred{ext}")
    print(colored(f"Processing {video_file}", TERM_COLOR))

    # Initialize Predictor
    predictor = GazePredictor(exp_path=EXP_PATH, device=DEVICE)
    print(colored(f"Using model from {EXP_PATH}", TERM_COLOR))

    # Read Video Clip
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    img_h, img_w, _ = frame.shape  # retrieve video height and width
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))//2
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read head tracks
    if args.head_tracks_file is not None:
        head_tracks = pd.read_csv(args.head_tracks_file)
        img_names = head_tracks['img'].values
        frame_nb = [int(img_name.split("_")[-1].split(".")[0]) for img_name in img_names]
        head_tracks['frame_nb'] = frame_nb
        head_tracks = head_tracks.groupby("frame_nb")
    
    # Initialize ffmpeg writer
    command = f"ffmpeg -loglevel error -y -s {img_w}x{img_h} -pixel_format rgb24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {output_file}"
    command = shlex.split(command)
    process = sp.Popen(command, stdin=sp.PIPE)
    
    # Iterate over frames and process
    predictions = []
    frame_nb = 0
    with tqdm(total=frame_count) as pbar:
        while ret:
            frame_nb += 1

            # =============== Get Head Bboxes =============== #
            head_bboxes = None; pids = None
            if args.head_tracks_file is not None:
                if frame_nb not in head_tracks.groups.keys():
                    process.stdin.write(frame.tobytes())
                    ret, frame = cap.read()
                    pbar.update(1)
                    continue
                head_tracks_frame = head_tracks.get_group(frame_nb)
                head_bboxes = torch.tensor(head_tracks_frame[["head_bbox_x_min", "head_bbox_y_min", "head_bbox_x_max", "head_bbox_y_max"]].values).float()
                valid_indices = torch.where(~torch.isnan(head_bboxes).any(dim=1))[0]
                head_bboxes = head_bboxes[valid_indices]
                pids = torch.tensor(head_tracks_frame["personId"].values).long()
                pids = pids[valid_indices]
            
            # =============== Predict =============== #
            depth = None #Image.open("/idiap/temp/stafasca/data/ChildPlay-depth/images/6mA6UAoT3M0_6165-6361/6mA6UAoT3M0_6168.png")
            frame = frame[..., ::-1] # BGR >> RGB
            image = Image.fromarray(frame)
            gaze_heatmaps, gaze_points, gaze_vecs, inouts, lah, laeo, coatt, head_bboxes, pids = predictor.predict(image, head_bboxes, pids, depth)
            gaze_heatmaps = gaze_heatmaps.cpu()
            gaze_points = gaze_points.cpu()
            gaze_vecs = gaze_vecs.cpu()
            inouts = inouts.sigmoid().cpu()
            lah = lah.sigmoid().cpu()
            laeo = laeo.sigmoid().cpu()
            coatt = coatt.sigmoid().cpu()
            head_bboxes = head_bboxes.cpu()
            num_people = len(head_bboxes)
            
            if len(head_bboxes) == 0:
                process.stdin.write(frame.tobytes())
                ret, frame = cap.read()
                pbar.update(1)
                continue

            # =============== Perform post-processing for LAH, LAEO ================ #    
            pair_indices = torch.tensor(list(itertools.permutations(torch.arange(num_people+1), 2)))
            lah = torch.zeros_like(lah); laeo = torch.zeros_like(laeo)
            for ppid, pair in enumerate(pair_indices):    # iterate over pairs
                # If padded person 
                if pair[0]==0 or pair[1]==0:
                    continue

                head_bbox1 = head_bboxes[pair[0]-1].tolist()
                gaze_pred1 = (gaze_points[pair[0]-1] * torch.tensor([img_w, img_h])).tolist()

                head_bbox2 = head_bboxes[pair[1]-1].tolist()
                gaze_pred2 = (gaze_points[pair[1]-1] * torch.tensor([img_w, img_h])).tolist()

                if is_inside(head_bbox1, gaze_pred2):
                    lah[0][ppid] = 1
                    if is_inside(head_bbox2, gaze_pred1):
                        laeo[0][ppid] = 1
            
            # =============== Process social gaze predictions =============== #
            social_preds = get_social_preds(lah, laeo, coatt, num_people+1)    # add 1 to account for padding
            
            # =============== Store Prediction =============== #
            prediction = {"frame_nb": frame_nb, 
                          "gaze_vecs": gaze_vecs, 
                          "gaze_points": gaze_points, 
                          "inouts": inouts, 
                          "pids": pids,
                          "head_bboxes": head_bboxes,
                          "lah": social_preds[0],
                          "laeo": social_preds[1],
                          "coatt": social_preds[2]}
            predictions.append(prediction)

            # =============== Draw Prediction =============== #
            num_people = len(head_bboxes)
            inouts = torch.ones((num_people, ))
            heatmap_pid = args.heatmap_pid if args.heatmap_pid >= 0 else None
    
            frame = draw_gaze(frame,
                              social_preds,
                              head_bboxes = head_bboxes, 
                              gaze_points = gaze_points, 
                              gaze_vecs = gaze_vecs if args.show_gaze_vec else None, 
                              inouts = inouts, 
                              pids = pids, 
                              gaze_heatmaps = gaze_heatmaps, 
                              heatmap_pid = heatmap_pid, 
                              frame_nb = frame_nb if args.show_frame_nb else None, 
                              exp_path = EXP_PATH if args.show_exp_path else None, 
                              alpha = args.alpha, 
                              gaze_pt_size = args.gaze_pt_size,
                              head_center_size = args.head_center_size,
                              thickness = args.thickness,
                              fs = args.fs,
                             ) 

            # ================= Write Frame ================= #
            process.stdin.write(frame.tobytes())
            
            # =============== Visualize Frame =============== #
            # Visualize live   
            #cv2.imshow('Frame', frame)
            #cv2.setWindowProperty('Frame', cv2.WND_PROP_TOPMOST, 1)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break 

            # =============== Read Next Frame =============== #
            ret, frame = cap.read()

            pbar.update(1)

    # Save predictions
    prediction_file = os.path.join(args.output_dir, f"{basename}-exp{exp}-pred.csv")
    save_predictions(predictions, prediction_file, img_w, img_h)

    # Reinitialize tracker
    predictor.tracker = predictor.init_tracker()
    
    # Closes all the opencv frames
    #cv2.destroyAllWindows()
    
    # Release Capture Device
    cap.release()
    
    # Close and flush stdin
    process.stdin.close()
    
    # Wait for sub-process to finish
    process.wait()
    
    # Terminate the sub-process
    process.terminate()
    
    end = dt.datetime.now()
    print(colored(f"Finished. The script took {end - start}.", TERM_COLOR))
        

if __name__ == "__main__":
    main()