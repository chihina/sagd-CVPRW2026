import math
from enum import Enum, auto
from typing import List, Tuple, Union
import itertools
import numpy as np
import pickle
import einops
import torch
import torch.nn.functional as F
from PIL import Image
import sys

def load_pkl(path_to_file):
    with open(path_to_file, 'rb') as file:
        return pickle.load(file)
    

def spherical2cartesian(gs):
    """
    Converts 2D spherical coordinates (yaw, pitch) or (theta, phi) into 3D cartesian coordinates (x, y, z).
    Args:
        gs: tensor (N, 2) or (2,) - coordinates in 2D spherical system
    Returns:
        gc: tensor (N, 3) or (3,) - coordinates in 3D cartesian system
    """
    device = gs.device
    
    if gs.dim() == 1:
        gs = gs.unsqueeze(0)
    gc = torch.empty(gs.size(0), 3, device=device)
    gc[:, 0] = torch.cos(gs[:, 1]) * torch.sin(gs[:, 0])
    gc[:, 1] = torch.sin(gs[:, 1])
    gc[:, 2] = -torch.cos(gs[:, 1]) * torch.cos(gs[:, 0])
    if gc.size(0) == 1:
        gc = gc.squeeze(0)
    
    return gc

def get_ptcloud(depthmap, focal_length):
    cx, cy, fx, fy, rm = get_K(depthmap.shape, focal_length)

    height, width = depthmap.shape[:2]

    space_DW = torch.linspace(0, width - 1, width, dtype=torch.float32)
    space_DH = torch.linspace(0, height - 1, height, dtype=torch.float32)
    space_yy, space_xx = torch.meshgrid(space_DH, space_DW, indexing='ij')

    space_X = (space_xx - cx) * depthmap / fx
    space_Y = (space_yy - cy) * depthmap / fy
    space_Z = depthmap

    points_3d = torch.stack([space_X, space_Y, space_Z], dim=-1)
    points_3d = points_3d.view(-1, 3)
    points_3d = torch.matmul(points_3d, rm.T)

    points_3d = points_3d.view(height, width, 3)

    return points_3d


def get_K(imgshape, focal_length):
    
    cx = imgshape[1]/2
    cy = imgshape[0]/2
    fx = fy = focal_length
    R=torch.eye(3, dtype=torch.float32)
    
    return [cx, cy, fx, fy, R]


def CameraToEyeMatrix(dirEyes, inout):
    if inout.dim()==0:
        inout=inout.unsqueeze(0)
    
    # Define left? hand coordinate system in the eye plane orthogonal to the camera ray
    cam2eye = []
    for de, io in zip(dirEyes, inout):
        if io!=-1:
            upVector = torch.tensor([0,-1,0], dtype=torch.float32)
            zAxis = de.flatten()
            xAxis = torch.cross(upVector, zAxis)
            xAxis /= torch.linalg.norm(xAxis)
            yAxis = torch.cross(zAxis, xAxis)
            yAxis /= torch.linalg.norm(yAxis) # not really necessary
            gazeCS = torch.stack([xAxis, yAxis, zAxis], axis=0)
        else:
            gazeCS = torch.eye(3, dtype=torch.float32)
        cam2eye.append(gazeCS)
    return torch.stack(cam2eye)


def generate_gaze_map(gaze_vecs, origin_pts, depth_maps, size=(224, 224), transform=None):
    """Function to generate a (batch) gaze cone from a given (batch) gaze vector and (batch) origin.

    Args:
        gaze_vecs (torch.Tensor | np.ndarray): gaze vector in format (B, 2) or (2,).
        origin_pts (torch.Tensor | np.ndarray): origin point in image coordinates (e.g. head center point) in format (B, 2) or (2,).
        size (tuple, optional): size of the canvas in the format [width, height]. Defaults to (224, 224).
        transform (Callable, optional): transformation to apply to the gaze cone(s). Defaults to None.

    Returns:
        torch.tensor | np.array: a gaze cone image where each non-filtered pixel represents the cosine
        of the angle between the gaze vector and the vector from the origin point to the pixel. Output format is either (B, H, W) or (H, W).
    """
    assert gaze_vecs.ndim == origin_pts.ndim, "Gaze point(s) and origin point(s) must have the same number of dimensions."

    if gaze_vecs.ndim == 1:
        gaze_vecs = gaze_vecs.unsqueeze(0)
        origin_pts = origin_pts.unsqueeze(0)

    assert (gaze_vecs != 0).any(1).all(), "Gaze vectors cannot be (0., 0.)."

    bs = len(gaze_vecs)
    device = gaze_vecs.device

    img_w, img_h = size
    xs, ys = torch.meshgrid(torch.arange(img_w), torch.arange(img_h), indexing="xy")
    xs, ys = xs.flatten().to(device), ys.flatten().to(device)
    mesh_vecs = torch.stack([xs.repeat(bs, 1) / img_w, ys.repeat(bs, 1) / img_h, depth_maps[:, 0, ys, xs]], dim=2) - origin_pts.unsqueeze(1)
    mesh_vecs = F.normalize(mesh_vecs, p=2, dim=2)

    gaze_vecs = F.normalize(gaze_vecs, p=2, dim=1)
    gaze_maps = (mesh_vecs @ gaze_vecs.unsqueeze(2)).squeeze(2)
    
    if transform is not None:
        gaze_maps = transform(gaze_maps)
    gaze_maps = gaze_maps.view(bs, img_h, img_w)

    if gaze_maps.shape[0] == 1:
        gaze_maps = gaze_maps.squeeze(0)

    return gaze_maps


def get_img_size(image):
    if isinstance(image, Image.Image):
        img_w, img_h = image.size
    elif isinstance(image, torch.Tensor):
        img_w, img_h = image.shape[2], image.shape[1]
    else:
        raise Exception(f"Input image needs to be either a Image.Image or torch.Tensor. Found {type(image)} instead.")
    return img_w, img_h


def pair(size):
    return size if isinstance(size, (list, tuple)) else (size, size)


class Stage(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    PREDICT = auto()


def parse_experiment(experiment: str):
    if "+" in experiment:
        steps = experiment.split("+")


def expand_bbox(bboxes, img_w, img_h, k=0.1):
    """
    Expand bounding boxes by a factor of k.

    Args:
        bboxes: a tensor of size (B, 4) or (4,) containing B boxes or a single box in the format [xmin, ymin, xmax, ymax]
        k: a scalar value indicating the expansion factor
        img_w: a scalar value indicating the width of the image
        img_h: a scalar value indicating the height of the image

    Returns:
        A tensor of size (B, 4) or (4,) containing the expanded bounding boxes in the format [xmin, ymin, xmax, ymax].
    """
    if len(bboxes.shape) == 1:
        bboxes = bboxes.unsqueeze(0)  # Add batch dimension if only a single box is provided

    # Compute the width and height of the bounding boxes
    bboxes_w = bboxes[:, 2] - bboxes[:, 0]
    bboxes_h = bboxes[:, 3] - bboxes[:, 1]

    # Compute expansion values
    expand_w = k * bboxes_w
    expand_h = k * bboxes_h

    # Expand the bounding boxes
    expanded_bboxes = torch.stack(
        [
            torch.clamp(bboxes[:, 0] - expand_w, min=0.0),
            torch.clamp(bboxes[:, 1] - expand_h, min=0.0),
            torch.clamp(bboxes[:, 2] + expand_w, max=img_w),
            torch.clamp(bboxes[:, 3] + expand_h, max=img_h),
        ],
        dim=1,
    )

    return expanded_bboxes.squeeze(0) if len(bboxes.shape) == 1 else expanded_bboxes


def square_bbox(bboxes, img_width, img_height):
    """
    Adjust bounding boxes to be squared while ensuring the center of the box doesn't change.
    If the bounding box is too close to the edge, recenter the box to keep it within the image frame.

    Args:
        bboxes: a tensor of size (B, 4) containing B bounding boxes in the format [xmin, ymin, xmax, ymax]
        img_width: a scalar value indicating the width of the image
        img_height: a scalar value indicating the height of the image

    Returns:
        A tensor of size (B, 4) containing the squared bounding boxes.
    """
    n = len(bboxes)
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    # Calculate original widths and heights
    widths = xmax - xmin
    heights = ymax - ymin

    # Calculate centers
    center_x = xmin + widths / 2
    center_y = ymin + heights / 2

    # Calculate maximum side length
    max_side_length = torch.max(widths, heights)

    # Calculate new xmin, ymin, xmax, ymax
    new_xmin = center_x - max_side_length / 2
    new_ymin = center_y - max_side_length / 2
    new_xmax = center_x + max_side_length / 2
    new_ymax = center_y + max_side_length / 2

    # Create the squared bounding boxes
    squared_bboxes = torch.stack([new_xmin, new_ymin, new_xmax, new_ymax], dim=1)

    return squared_bboxes


def gaussian_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    mx: Union[float, torch.Tensor] = 0.0,
    my: Union[float, torch.Tensor] = 0.0,
    sx: float = 1.0,
    sy: float = 1.0,
):
    out = 1 / (2 * math.pi * sx * sy) * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    return out


def generate_gaze_heatmap(gaze_pts: torch.Tensor, sigma: Union[int, Tuple] = 3, size: Union[int, Tuple] = 64) -> torch.Tensor:
    """
    Function to generate a gaze heatmap from a set of gaze points. Every pixel beyond 3 standard deviations
    from the gaze point is set to 0.

    Args:
        gaze_pts (torch.Tensor): normalized gaze points (ie. [num_heads, gaze_x, gaze_y]) between [0, 1].
        sigma (Union[int, Tuple], optional): standard deviation. Defaults to 3.
        size (Union[int, Tuple], optional): spatial size of the output (ie. [width, height]). Defaults to 64.

    Returns:
        torch.Tensor: the gaze heatmap corresponding to gaze_pt
    """

    device = gaze_pts.device
    size = torch.tensor((size, size)) if isinstance(size, int) else torch.tensor(size)
    sigma = torch.tensor((sigma, sigma)) if isinstance(sigma, int) else torch.tensor(sigma)
    gaze_pts = gaze_pts * size
    num_heads = len(gaze_pts)

    heatmaps = torch.zeros((num_heads, size[1], size[0]), dtype=torch.float, device=device)
    x = torch.arange(0, size[0], device=device)
    y = torch.arange(0, size[1], device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')
    for hi, gp in enumerate(gaze_pts):
        if gp[1]>0:
            heatmaps[hi] = gaussian_2d(x, y, gp[0], gp[1], sigma[0], sigma[1])
            heatmaps[hi] /= heatmaps[hi].max()

    return heatmaps

def generate_coatt_heatmap(gaze_hms: torch.Tensor, coatt_ids: torch.Tensor, num_coatt: int, size: Union[int, Tuple] = 64) -> torch.Tensor:
    """
    Function to generate a coatt heatmap from a set of gaze heatmaps. 

    Args:
        gaze_hms (torch.Tensor): gaze heatmaps of each person
        size (Union[int, Tuple], optional): spatial size of the output (ie. [width, height]). Defaults to 64.

    Returns:
        torch.Tensor: the coatt heatmaps
    """

    device = gaze_hms.device
    coatt_id_set = set(coatt_ids.tolist())

    # remove 0 from coatt_id_set because it is a padded id
    if 0 in coatt_id_set:
        coatt_id_set.remove(0)

    heatmaps = []
    coatt_levels = []
    for coatt_id in coatt_id_set:
        coatt_people_indices = torch.where(coatt_ids==coatt_id)
        gaze_hms_coatt = gaze_hms[coatt_people_indices]
        gaze_hms_coatt_mean = gaze_hms_coatt.mean(dim=0)
        heatmaps.append(gaze_hms_coatt_mean)
        coatt_level = torch.zeros(len(coatt_ids), dtype=torch.int, device=device)
        coatt_level[coatt_people_indices] = 1
        coatt_levels.append(coatt_level)

    if len(heatmaps) > 0:
        heatmaps = torch.stack(heatmaps)
        heatmaps_pad = torch.zeros((num_coatt-heatmaps.shape[0], size, size), dtype=torch.float, device=device)
        heatmaps = torch.cat([heatmaps, heatmaps_pad], dim=0)
        coatt_levels = torch.stack(coatt_levels)
        coatt_levels_pad = torch.zeros((num_coatt-coatt_levels.shape[0], len(coatt_ids)), dtype=torch.int, device=device)
        coatt_levels = torch.cat([coatt_levels, coatt_levels_pad], dim=0)
    else:
        heatmaps = torch.zeros((num_coatt, size, size), dtype=torch.float, device=device)
        coatt_levels = torch.zeros((num_coatt, len(coatt_ids)), dtype=torch.int, device=device)

    return heatmaps, coatt_levels
    

def generate_mask(bboxes, img_w, img_h):
    """
    Create a binary mask tensor where pixels inside the bounding boxes have a value of 1.

    Args:
        bboxes: a tensor of size (N, 4) or (4,) containing N or 1 bounding boxes in the format [xmin, ymin, xmax, ymax]
                normalized to [0, 1]
        img_w: a scalar value indicating the width of the image
        img_h: a scalar value indicating the height of the image

    Returns:
        A binary tensor of shape (N, 1, img_height, img_width) where pixels inside the bounding boxes
        have a value of 1.
    """

    ndim = bboxes.ndim
    if ndim == 1:
        bboxes = bboxes.unsqueeze(0)

    # Calculate pixel coordinates of bounding boxes
    xmin = (bboxes[:, 0] * img_w).long()
    ymin = (bboxes[:, 1] * img_h).long()
    xmax = (bboxes[:, 2] * img_w).long()
    ymax = (bboxes[:, 3] * img_h).long()

    # Determine the number of boxes
    num_boxes = bboxes.shape[0]

    # Create empty binary mask tensor
    mask = torch.zeros((num_boxes, 1, img_h, img_w), dtype=torch.float32, device=bboxes.device)

    # Generate grid of indices
    grid_y, grid_x = torch.meshgrid(
        torch.arange(img_h, device=bboxes.device),
        torch.arange(img_w, device=bboxes.device),
        indexing='ij'
    )

    # Reshape grid indices for broadcasting
    grid_y = grid_y.view(1, img_h, img_w)
    grid_x = grid_x.view(1, img_h, img_w)

    # Determine if each pixel falls within any of the bounding boxes
    inside_mask = (grid_x >= xmin.view(num_boxes, 1, 1)) & (grid_x <= xmax.view(num_boxes, 1, 1)) & (grid_y >= ymin.view(num_boxes, 1, 1)) & (grid_y <= ymax.view(num_boxes, 1, 1))

    # Set corresponding pixels to 1 in the mask tensor
    mask[inside_mask.unsqueeze(1)] = 1
    return mask.squeeze(0) if ndim == 1 else mask


def generate_gaze_cone(gaze_vecs, origin_pts, size=(224, 224), thr=90, transform=None):
    """Function to generate a (batch) gaze cone from a given (batch) gaze vector and (batch) origin.

    Args:
        gaze_vecs (torch.Tensor | np.ndarray): gaze vector in format (B, 2) or (2,).
        origin_pts (torch.Tensor | np.ndarray): origin point in image coordinates (e.g. head center point) in format (B, 2) or (2,).
        size (tuple, optional): size of the canvas in the format [width, height]. Defaults to (224, 224).
        thr (int, optional): threshold angle in degrees to filter out from the gaze cone. Defaults to 90.
        transform (Callable, optional): transformation to apply to the gaze cone(s). Defaults to None.

    Returns:
        torch.tensor | np.array: a gaze cone image where each non-filtered pixel represents the cosine
        of the angle between the gaze vector and the vector from the origin point to the pixel. Output format is either (B, H, W) or (H, W).
    """
    assert gaze_vecs.ndim == origin_pts.ndim, "Gaze point(s) and origin point(s) must have the same number of dimensions."

    if gaze_vecs.ndim == 1:
        gaze_vecs = gaze_vecs.unsqueeze(0)
        origin_pts = origin_pts.unsqueeze(0)

    assert (gaze_vecs != 0).any(1).all(), "Gaze vectors cannot be (0., 0.)."

    bs = len(gaze_vecs)
    device = gaze_vecs.device

    thr = math.cos(thr * math.pi / 180)

    img_w, img_h = size
    xs, ys = torch.meshgrid(torch.arange(img_w), torch.arange(img_h), indexing="xy")
    xs, ys = xs.flatten().to(device), ys.flatten().to(device)

    gaze_vecs = F.normalize(gaze_vecs, p=2, dim=1)

    mesh_vecs = torch.stack([xs, ys], dim=1).repeat(bs, 1, 1) - (origin_pts * torch.tensor([img_w, img_h], device=device)).unsqueeze(1)
    mesh_vecs = F.normalize(mesh_vecs, p=2, dim=2)

    gaze_cones = (mesh_vecs @ gaze_vecs.unsqueeze(2)).squeeze(2)
    gaze_cones[gaze_cones < thr] = 0
    if transform is not None:
        gaze_cones = transform(gaze_cones)
    gaze_cones = gaze_cones.view(bs, img_h, img_w)

    if gaze_cones.shape[0] == 1:
        gaze_cones = gaze_cones.squeeze(0)

    return gaze_cones


def spatial_argmax2d(heatmap, normalize=True):
    """
    Function to locate the coordinates of the max value in the heatmap.
    Computation is done under no_grad() context.

    Args:
        heatmap (torch.Tensor): The input heatmap of shape (H, W) or (B, H, W).
        normalize (bool, optional): Specifies whether to normalize the argmax coordinates to [0, 1]. Defaults to True.

    Returns:
        torch.Tensor: The (normalized) argmax coordinates in the form (x, y) (i.e. shape (B, 2) or (2,))
    """

    with torch.no_grad():
        ndim = heatmap.ndim
        if ndim == 2:
            heatmap = heatmap.unsqueeze(0)

        points = (heatmap == torch.amax(heatmap, dim=(1, 2), keepdim=True)).nonzero()
        points = remove_duplicate_max(points)
        points = points[:, 1:].flip(1)  # (idx, y, x) -> (x, y)

        if normalize:
            points = points / torch.tensor(heatmap.size()[1:]).flip(0).to(heatmap.device)

        if ndim == 2:
            points = points[0]

    return points


def remove_duplicate_max(pts):
    """
    Function to remove duplicate rows based on the values of the first column (i.e. representing indices).
    The first occurence of each index value is kept.

    Args:
        pts (torch.Tensor): The points tensor of shape (N, 3) where 3 represents (index, y, x).

    Returns:
        torch.Tensor: Tensor of shape (M, 3) where M <= N after removing duplicates based on index value.
    """
    _, counts = torch.unique_consecutive(pts[:, 0], return_counts=True, dim=0)
    cum_sum = counts.cumsum(0)
    first_unique_idx = torch.cat((torch.tensor([0], device=pts.device), cum_sum[:-1]))
    return pts[first_unique_idx]


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings from MoCo-v3

    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

    assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])

    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = einops.rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w, d=embed_dim)

    return pos_emb


def generate_binary_gaze_heatmap(gaze_point, size=(64, 64)):
    """Draw the gaze point(s) on an empty canvas to produce a binary heatmap,
    where the location(s) of the gaze point(s) correspond to 1 while the rest
    is set to 0.

    Args:
        gaze_point (torch.Tensor): Gaze point(s) to draw.
        size (tuple, optional): Size of the output image [height, width]. Defaults to (64, 64).

    Returns:
        torch.Tensor: A binary gaze heatmap.
    """
    assert gaze_point.ndim <= 2, f"Gaze point must be 1D or 2D, but found {gaze_point.ndim}D."

    height, width = size
    gaze_point = gaze_point * (torch.tensor((width, height), device=gaze_point.device) - 1)
    gaze_point = gaze_point.int()
    binary_heatmap = torch.zeros((height, width), device=gaze_point.device, dtype=torch.int)

    if gaze_point.ndim == 1:
        binary_heatmap[gaze_point[1], gaze_point[0]] = 1
    elif gaze_point.ndim == 2:  # gazefollow
        for gp in gaze_point:
            binary_heatmap[gp[1], gp[0]] = 1

    return binary_heatmap


def is_coatt(coatt1, coatt2):
    return (coatt1 == coatt2) & (coatt1 != -1)

# function to get pairwise SA labels
# 0: padded head
def id_to_pairwise_coatt(coatt_ids):

    if coatt_ids.ndim == 1:
        coatt_ids = coatt_ids.unsqueeze(0)
    
    batch_size, num_people = coatt_ids.shape
#     indices = torch.triu_indices(num_people, num_people, offset=1).T
    indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
    num_pairs = len(indices)
    
    pairwise_coatt = torch.empty((batch_size, num_pairs), device=coatt_ids.device)
    mask = torch.zeros((batch_size, num_pairs), device=coatt_ids.device, dtype=bool)
    
    for k in range(len(indices)):
        i = indices[k, 0]
        j = indices[k, 1]
        pairwise_coatt[:, k] = is_coatt(coatt_ids[:, i], coatt_ids[:, j])
        # mask[:, k] = (coatt_ids[:, i] != 0) & (coatt_ids[:, j] != 0) & ((coatt_ids[:,i]!=-100) | (coatt_ids[:,j]!=-100))
        mask[:, k] = ((coatt_ids[:,i]!=-100) | (coatt_ids[:,j]!=-100))
    
    return pairwise_coatt, mask

def is_coatt_vectorized(ids_i, ids_j):
    """
    A vectorized version of the is_coatt logic.
    Assumes co-attention if IDs are the same and are valid (e.g., > 0).
    -100 is treated as an invalid person ID.
    """
    # People are in co-attention if they have the same ID, and that ID is not 0 or -100
    # Adjust this logic based on your specific definition of a valid co-attention ID.
    return (ids_i == ids_j) & (ids_i != -1)

def id_to_pairwise_coatt_vectorized(coatt_ids):
    """
    A vectorized version of id_to_pairwise_coatt that avoids Python for-loops
    for significant speedup.
    """
    if coatt_ids.ndim == 1:
        coatt_ids = coatt_ids.unsqueeze(0)
    
    batch_size, num_people = coatt_ids.shape

    # If there's only one person, no pairs can be formed.
    if num_people < 2:
        return torch.empty((batch_size, 0), device=coatt_ids.device), \
               torch.empty((batch_size, 0), device=coatt_ids.device, dtype=bool)

    # 1. Generate all pair indices at once
    # Use .to(device) to ensure indices are on the same device as the input
    indices = torch.tensor(list(itertools.permutations(range(num_people), 2)), device=coatt_ids.device)
    i_indices = indices[:, 0]
    j_indices = indices[:, 1]

    # 2. Gather the coatt_ids for all 'i' and 'j' pairs across the batch in one go
    # Shape of both ids_i and ids_j will be (batch_size, num_pairs)
    ids_i = coatt_ids[:, i_indices]
    ids_j = coatt_ids[:, j_indices]

    # 3. Apply the co-attention and mask logic to all pairs simultaneously (vectorized)
    pairwise_coatt = is_coatt_vectorized(ids_i, ids_j).float() # Use .float() to convert boolean to 0.0s and 1.0s
    mask = (ids_i != -100) | (ids_j != -100)
    
    return pairwise_coatt, mask

# function to get pairwise LAEO labels
# 0: padded head; -100: not annotated head
def id_to_pairwise_laeo(coatt_ids):
    
    if coatt_ids.ndim == 1:
        coatt_ids = coatt_ids.unsqueeze(0)
    
    batch_size, num_people = coatt_ids.shape
#     indices = torch.triu_indices(num_people, num_people, offset=1).T
    indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
    num_pairs = len(indices)
    
    pairwise_coatt = torch.empty((batch_size, num_pairs), device=coatt_ids.device)
    mask = torch.zeros((batch_size, num_pairs), device=coatt_ids.device, dtype=bool)
    
    for k in range(len(indices)):
        i = indices[k, 0]
        j = indices[k, 1]
        pairwise_coatt[:, k] = is_coatt(coatt_ids[:, i], coatt_ids[:, j])
        mask[:, k] = (coatt_ids[:, i] != 0) & (coatt_ids[:, j] != 0) & ((coatt_ids[:,i]>0) | (coatt_ids[:,j]>0)) & ((coatt_ids[:,i]+coatt_ids[:,j])!=0)
        
    return pairwise_coatt, mask

def id_to_pairwise_laeo_vectorized(coatt_ids):
    if coatt_ids.ndim == 1:
        coatt_ids = coatt_ids.unsqueeze(0)
    
    batch_size, num_people = coatt_ids.shape

    if num_people < 2:
        return torch.empty((batch_size, 0), device=coatt_ids.device), \
               torch.empty((batch_size, 0), device=coatt_ids.device, dtype=bool)

    indices = torch.tensor(list(itertools.permutations(range(num_people), 2)), device=coatt_ids.device)
    i_indices = indices[:, 0]
    j_indices = indices[:, 1]

    ids_i = coatt_ids[:, i_indices]  # shape: (batch_size, num_pairs)
    ids_j = coatt_ids[:, j_indices]  # shape: (batch_size, num_pairs)

    pairwise_coatt = is_coatt_vectorized(ids_i, ids_j).float()
    
    mask = (ids_i != 0) & \
           (ids_j != 0) & \
           ((ids_i > 0) | (ids_j > 0)) & \
           ((ids_i + ids_j) != 0)
    
    return pairwise_coatt, mask

# function to get pairwise LAH labels
# -5: valid head, unknown lah label, -3: padded head, -1: not lah, >=0: lah head index
def id_to_pairwise_lah(lah_ids):
    
    if lah_ids.ndim == 1:
        lah_ids = lah_ids.unsqueeze(0)
    
    batch_size, num_people = lah_ids.shape
    if num_people>1:
        indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
    else:
        indices = torch.zeros(2, dtype=torch.long).unsqueeze(0)
    num_pairs = len(indices)
    
    pairwise_lah = torch.empty((batch_size, num_pairs), device=lah_ids.device)
    mask = torch.zeros((batch_size, num_pairs), device=lah_ids.device, dtype=bool)
    
    for k in range(len(indices)):
        i = indices[k, 0]
        j = indices[k, 1]
        pairwise_lah[:, k] = (lah_ids[:, j]==i)
        mask[:, k] = (lah_ids[:, j] != -5) & (lah_ids[:, j] != -3) #& (lah_ids[:, i] != -3) 
        
    return pairwise_lah, mask

def id_to_pairwise_lah_vectorized(lah_ids):
    if lah_ids.ndim == 1:
        lah_ids = lah_ids.unsqueeze(0)
    
    batch_size, num_people = lah_ids.shape

    if num_people < 2:
        return torch.empty((batch_size, 0), device=lah_ids.device), \
               torch.empty((batch_size, 0), device=lah_ids.device, dtype=bool)

    indices = torch.tensor(list(itertools.permutations(range(num_people), 2)), device=lah_ids.device)
    i_indices = indices[:, 0]
    j_indices = indices[:, 1]

    ids_j = lah_ids[:, j_indices]  # shape: (batch_size, num_pairs)

    pairwise_lah = (ids_j == i_indices).float()
    
    mask = (ids_j != -5) & (ids_j != -3)
    
    return pairwise_lah, mask

# function to convert laeo ids to lah ids
def laeo2lah(laeo_ids):
    
    lah_ids = torch.zeros_like(laeo_ids) - 5
    for i, lid in enumerate(laeo_ids):
        if lid==0 or lid==-100:
            continue
        match_ids = torch.where(laeo_ids[i+1:]==lid)[0]
        if len(match_ids)==1:
            lah_ids[i] = i+1+match_ids.item()
            lah_ids[i+1+match_ids.item()] = i
            
    return lah_ids       

# function to convert lah ids to laeo ids
def lah2laeo(lah_ids):
    
    laeo_ids = torch.zeros_like(lah_ids) - 100    # special id to indicate not annotated heads
    laeo_ids[lah_ids==-3] = 0   # ignore padded heads
    for i, lid in enumerate(lah_ids):
        if lid==-5 or lid==-3:
            continue
        laeo_ids[i] = i+1
        if lid!=-1:
            if lah_ids[lid]==i:
                laeo_ids[lid] = i+1    # provide matching id for LAEO pairs
            elif lah_ids[lid]==-5:
                laeo_ids[lid] = -i-1    # special ids to ignore this pair
    
    return laeo_ids

# function to convert lah ids to coatt ids
def lah2coatt(lah_ids):
    
    coatt_ids = torch.zeros_like(lah_ids)    
    for i, lid in enumerate(lah_ids):
        if lid==-5 or lid==-3:
            continue
        if lid!=-1:
            match_ids = torch.where(lah_ids==lid)[0]
            coatt_ids[match_ids] = i+1    # provide matching id for coatt people
        else:
            coatt_ids[i] = -100   # special id for not lah cases
    
    return coatt_ids