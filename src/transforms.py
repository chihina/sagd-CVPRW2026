from typing import List, Tuple
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF

# IMPORTS
from PIL import Image



# ============================================================================= #
#                                     UTILITY                                   #
# ============================================================================= #
def get_img_size(image):
    if isinstance(image, Image.Image):
        img_w, img_h = image.size
    elif isinstance(image, torch.Tensor):
        img_w, img_h = image.shape[2], image.shape[1]
    else:
        raise Exception(f"Input image needs to be either a Image.Image or torch.Tensor. Found {type(image)} instead.")
    return img_w, img_h


def pair(size):
    return size if isinstance(size, tuple) else (size, size)




# ============================================================================= #
#                                    TRANSFORMS                                 #
# ============================================================================= #
class RandomHeadBboxJitter(object):
    """
    Applies random geometric transformations (ie. translations and/or expansions)
    to the head bounding box.
    """

    def __init__(self, p=1.0, tr=(-0.1, 0.1)):
        """
        Args:
            p (float, optional): probability of jittering the box. Defaults to 1.0.
            tr (tuple, optional): factor by which to increase the box in all directions. Defaults to (-0.1, 0.1).
        """
        self.p = p
        self.tr = tr if isinstance(tr, tuple) else (-abs(tr), abs(tr))

    def __call__(self, head_bboxes, img_w, img_h):
        
        if len(head_bboxes)==0:
            return head_bboxes
        
        if torch.rand(1) <= self.p:
            ws, hs = head_bboxes[:, [2]] - head_bboxes[:, [0]], head_bboxes[:, [3]] - head_bboxes[:, [1]]
            jitter = torch.empty((len(head_bboxes), 4)).uniform_(self.tr[0], self.tr[1])
            head_bboxes = head_bboxes + torch.cat([-ws, -hs, ws, hs], dim=1) * jitter
            head_bboxes[:, [0, 2]] = head_bboxes[:, [0, 2]].clip(0., img_w)
            head_bboxes[:, [1, 3]] = head_bboxes[:, [1, 3]].clip(0., img_h)
            
        return head_bboxes
    

# class Resize(object):
#     """
#     Resizes the input image to the desired shape.
#     """

#     def __init__(self, img_size, head_size):
#         assert isinstance(img_size, (int, tuple)), f"img_size needs to be either an int or tuple. Found {img_size} instead."
#         assert isinstance(head_size, tuple), f"head_size needs to be a tuple. Found {head_size} instead."
#         self.img_size = img_size
#         self.head_size = head_size


#     def __call__(self, sample):
#         num_heads = len(sample["heads"]) 
#         img_w, img_h = sample["image"].size
        
#         # Resize Image
#         sample["image"] = TF.resize(sample["image"], self.img_size)  # type: ignore
#         new_img_w, new_img_h = sample["image"].size
        
#         # Resize Heads
#         for k in range(num_heads):
#             sample["heads"][k] = TF.resize(sample["heads"][k], self.head_size)

#         return sample
    

# New Resize function, maintains aspect ratio during evaluation
class Resize(object):
    """
    Resizes the input image to the desired shape.
    """

    def __init__(self, img_size, head_size):
        assert isinstance(img_size, (int, tuple, list))
        assert isinstance(head_size, (int, tuple, list))
        self.img_size = img_size
        self.head_size = (
            head_size if isinstance(head_size, (tuple, list)) else (head_size, head_size)
        )

    def __call__(self, sample):    
        img_w, img_h = sample["image"].size
        
        if isinstance(self.img_size, (tuple,list)):
            t_image = TF.resize(sample["image"], (self.img_size[1], self.img_size[0]), antialias=True)  # type: ignore
            t_pcd = TF.resize(sample["pcd"], (self.img_size[1], self.img_size[0]), antialias=True)  # type: ignore
        else: # if a single number, resize smallest edge to size, ensure larger edge is multiple of 32
            
            if img_w < img_h:
                new_w = self.img_size
                new_h = int(img_h * new_w / img_w)
                q, r = divmod(new_h, 32)
                new_h = new_h if q == 0 else 32 * (q + 1)
            else:
                new_h = self.img_size
                new_w = int(img_w * new_h / img_h)
                q, r = divmod(new_w, 32)
                new_w = new_w if q == 0 else 32 * (q + 1)
                
            t_image = TF.resize(sample["image"], (new_h, new_w), antialias=True)
            t_pcd = TF.resize(sample["pcd"], (new_h, new_w), antialias=True)
        
        t_heads = []
        for head in sample["heads"]:
            t_head = TF.resize(head, self.head_size, antialias=True)
            t_heads.append(t_head)
        
        sample["image"] = t_image
        sample["pcd"] = t_pcd
        sample["heads"] = t_heads

        return sample
    

class RandomHorizontalFlip(object):
    """
    Flips the input image horizontally.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):

        if torch.rand(1) <= self.p:
            
            num_heads = len(sample["heads"])
            img_w, img_h = get_img_size(sample["image"])

            # Flip Image
            sample["image"] = TF.hflip(sample["image"])  # type: ignore
            
            # Flip Pcd
            sample["pcd"] = TF.hflip(sample["pcd"])  # type: ignore
            
            # Flip Heads
            for k in range(num_heads):
                sample["heads"][k] = TF.hflip(sample["heads"][k])
            
            if num_heads>0:
                # Flip Gaze points and Head Centers and Head Bboxes
                sample["gaze_pts"][sample["gaze_pts"][:, 0] != -1., 0] = 1.0 - sample["gaze_pts"][sample["gaze_pts"][:, 0] != -1., 0]
                valid_indices = torch.where(sample["head_bboxes"].sum(-1) != 0)[0]
                tmp = sample["head_bboxes"][valid_indices, 0].clone()
                sample["head_bboxes"][valid_indices, 0] = 1. - sample["head_bboxes"][valid_indices, 2]
                sample["head_bboxes"][valid_indices, 2] = 1. - tmp

        return sample
    

class ColorJitter(object):
    """
    Applies random colors transformations to the input (ie. brightness,
    contrast, saturation and hue).
    """

    def __init__(self, brightness, contrast, saturation, hue, p=1.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, sample):

        if torch.rand(1) <= self.p:
            num_heads = len(sample["heads"])

            # Sample color transformation factors and order
            brightness_factor = None if self.brightness is None else torch.rand(1).uniform_(*self.brightness)
            contrast_factor = None if self.contrast is None else torch.rand(1).uniform_(*self.contrast)
            saturation_factor = None if self.saturation is None else torch.rand(1).uniform_(*self.saturation)
            hue_factor = None if self.hue is None else torch.rand(1).uniform_(*self.hue)
            
            if np.array(sample['image']).sum()>0:
                fn_indices = torch.randperm(4)
                for fn_id in fn_indices:
                    if fn_id == 0 and brightness_factor is not None:
                        sample["image"] = TF.adjust_brightness(sample["image"], brightness_factor)                    
                        for k in range(num_heads):
                            if sample['head_bboxes'][k].sum()>0:
                                sample["heads"][k] = TF.adjust_brightness(sample["heads"][k], brightness_factor)

                    elif fn_id == 1 and contrast_factor is not None:
                        sample["image"] = TF.adjust_contrast(sample["image"], contrast_factor)                    
                        for k in range(num_heads):
                            if sample['head_bboxes'][k].sum()>0:
                                sample["heads"][k] = TF.adjust_contrast(sample["heads"][k], contrast_factor)

                    elif fn_id == 2 and saturation_factor is not None:
                        sample["image"] = TF.adjust_saturation(sample["image"], saturation_factor)                    
                        for k in range(num_heads):
                            if sample['head_bboxes'][k].sum()>0:
                                sample["heads"][k] = TF.adjust_saturation(sample["heads"][k], saturation_factor)

                    elif fn_id == 3 and hue_factor is not None:
                        sample["image"] = TF.adjust_hue(sample["image"], hue_factor)                    
                        for k in range(num_heads):
                            if sample['head_bboxes'][k].sum()>0:
                                sample["heads"][k] = TF.adjust_hue(sample["heads"][k], hue_factor)

        return sample
    

class Normalize(object):
    def __init__(self, img_mean=[0.44232, 0.40506, 0.36457], img_std=[0.28674, 0.27776, 0.27995]):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, sample):
        sample["image"] = TF.normalize(sample["image"], self.img_mean, self.img_std)
        num_heads = len(sample['heads'])
        if num_heads>0:
            sample["heads"] = TF.normalize(sample["heads"], self.img_mean, self.img_std)
        return sample


class ToTensor(object):
    """
    Convert inputs to tensors.
    """
    def __call__(self, sample):
        num_heads = len(sample["heads"])
        
        sample["image"] = TF.to_tensor(sample["image"])
        for k in range(num_heads):
            sample["heads"][k] = TF.to_tensor(sample["heads"][k])
        
        if num_heads>0:
            sample["heads"] = torch.stack(sample["heads"], dim=0)
        else:
            sample['heads'] = torch.tensor(sample["heads"])

        return sample


class RandomCropSafeGaze(object):
    """
    Randomly crops the input image while ensuring the gaze target and the head bounding box
    remain within the crop. The crop is also chosen such that it respects the given aspect
    ratio (or the aspect ratio of the image if None).
    """

    def __init__(self, aspect=None, p=0.8):
        self.aspect = aspect
        self.p = p

    def __call__(self, sample):

        if torch.rand(1) <= self.p:
            num_heads = len(sample["heads"])
            img_w, img_h = sample["image"].size
            
            # Randomly Sample Crop Area (in pixels, not [0, 1])
            crop_xmin, crop_ymin, crop_w, crop_h = self._get_random_crop_bbox(img_w, img_h, sample["head_bboxes"], sample["gaze_pts"])

            # Crop Image
            sample["image"] = TF.crop(sample["image"], int(crop_ymin), int(crop_xmin), int(crop_h), int(crop_w))
            
            # Crop Pcd
            sample["pcd"] = TF.crop(sample["pcd"], int(crop_ymin), int(crop_xmin), int(crop_h), int(crop_w))
            
            if num_heads>0:
                # Convert Head Bboxes
                sample["head_bboxes"] = sample["head_bboxes"] * torch.tensor([[img_w, img_h]]).repeat(1, 2)
                sample["head_bboxes"] = sample["head_bboxes"] - torch.tensor([[crop_xmin, crop_ymin]]).repeat(1, 2)
                sample["head_bboxes"] = sample["head_bboxes"] / torch.tensor([[crop_w, crop_h]]).repeat(1, 2)

                # Convert Gaze Points
                mask = (sample["gaze_pts"][:, 0] != -1.)
                sample["gaze_pts"][mask] = sample["gaze_pts"][mask] * torch.tensor([[img_w, img_h]]) - torch.tensor([[crop_xmin, crop_ymin]])
                sample["gaze_pts"][mask] = sample["gaze_pts"][mask] / torch.tensor([[crop_w, crop_h]])   
            
           
        return sample

    def _get_random_crop_bbox(self, img_w, img_h, head_bboxes, gaze_pts):
        """
        Computes the parameters of a random crop that maintains the aspect ratio, and includes
        the gaze point and head bounding box.
        """

        # Compute aspect ratio
        aspect = img_w / img_h if self.aspect is None else self.aspect
        
        coords = head_bboxes
        if len(gaze_pts)>0:
            coords = torch.concat([head_bboxes, gaze_pts[gaze_pts[:, 0] != -1.].repeat(1, 2)], dim=0)
        if len(coords)>0:
            zone_xmin = coords[:, 0].min().item() * img_w
            zone_ymin = coords[:, 1].min().item() * img_h
            zone_xmax = coords[:, 2].max().item() * img_w
            zone_ymax = coords[:, 3].max().item() * img_h
            if zone_xmax==0 or zone_ymax==0:
                return 0, 0, img_w, img_h    # no cropping if no head detections
        else:
            return 0, 0, img_w, img_h    # no cropping if no annotations

        # Expand the "safe" zone a bit (to include more image context - e.g. full object instead of just the point annotated)
        zone_xmin, zone_ymin, zone_xmax, zone_ymax = self._expand(zone_xmin, zone_ymin, zone_xmax, zone_ymax, img_w, img_h)
        zone_w = zone_xmax - zone_xmin
        zone_h = zone_ymax - zone_ymin

        # Randomly select a crop size
        if zone_w >= zone_h * aspect:
            crop_w = torch.rand(1).uniform_(zone_w, img_w).item()
            crop_h = crop_w / aspect
        else:
            crop_h = torch.rand(1).uniform_(zone_h, img_h).item()
            crop_w = crop_h * aspect

        # Find min and max possible positions for top-left point
        xmin = max(zone_xmax - crop_w, 0)
        xmax = min(zone_xmin, max(img_w - crop_w, 0)) # crop_w can be >= img_w
        ymin = max(zone_ymax - crop_h, 0)
        ymax = min(zone_ymin, max(img_h - crop_h, 0)) # crop_h can be >= img_h

        # Randomly select a top left point
        if xmin <= xmax:
            crop_xmin = torch.rand(1).uniform_(xmin, xmax).item()
        else:
            crop_xmin = 0.
            print(f"CAUGHT ERROR: xmin > xmax \n\n")
            print(f"(img_w, img_h) = ({img_w}, {img_h})")
            print(f"head_bboxes = {head_bboxes}")
            print(f"gaze_pts = {gaze_pts}")
            print(f"(xmin, ymin, xmax, ymax) = ({xmin}, {ymin}, {xmax}, {ymax})")
            print(f"(zone_xmin, zone_ymin, zone_xmax, zone_ymax) = ({zone_xmin}, {zone_ymin}, {zone_xmax}, {zone_ymax})")
            print(f"(zone_h, zone_w) = ({zone_h}, {zone_w})")
            print(f"(crop_h, crop_w) = ({crop_h}, {crop_w})")
            
        if ymin <= ymax:
            crop_ymin = torch.rand(1).uniform_(ymin, ymax).item()
        else:
            crop_ymin = 0.
            print(f"CAUGHT ERROR: ymin > ymax \n\n")
            print(f"(img_w, img_h) = ({img_w}, {img_h})")
            print(f"head_bboxes = {head_bboxes}")
            print(f"gaze_pts = {gaze_pts}")
            print(f"(xmin, ymin, xmax, ymax) = ({xmin}, {ymin}, {xmax}, {ymax})")
            print(f"(zone_xmin, zone_ymin, zone_xmax, zone_ymax) = ({zone_xmin}, {zone_ymin}, {zone_xmax}, {zone_ymax})")
            print(f"(zone_h, zone_w) = ({zone_h}, {zone_w})")
            print(f"(crop_h, crop_w) = ({crop_h}, {crop_w})")
            

        return crop_xmin, crop_ymin, crop_w, crop_h

    def _expand(self, xmin, ymin, xmax, ymax, img_w, img_h, k=0.2):
        # Expand bbox while ensuring it stays within image
        w, h = abs(xmax - xmin), abs(ymax - ymin)
        xmin = max(xmin - k * w, 0.0)
        ymin = max(ymin - k * h, 0.0)
        xmax = min(xmax + k * w, img_w)
        ymax = min(ymax + k * h, img_h)
        return xmin, ymin, xmax, ymax


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)

        return sample
