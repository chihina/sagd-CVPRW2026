# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from src.datasets import GazeFollowDataset
from src.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCropSafeGaze,
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

root = (
    "/Users/stafasca/Dropbox/Mac/Documents/Idiap/AI4Autism/Datasets/gazefollow_extended"
)
split = "train"
transform = Compose(
    [
        RandomCropSafeGaze(aspect=None, p=1.0),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(
            brightness=(0.5, 1.5),
            contrast=(0.5, 1.5),
            saturation=(0, 1.5),
            hue=None,
            p=0.8,
        ),
        Resize(img_size=(224, 224), head_size=(224, 224)),
        ToTensor(),
        Normalize(),
    ]
)
dataset = GazeFollowDataset(root, split, transform)

# %%
k = np.random.choice(len(dataset))  # 4769
sample = dataset[k]

#%%

fig, axes = plt.subplots(nrows=25, ncols=5, figsize=(30, 150))
axes = axes.flatten()

for i in range(len(axes)):
    sample = dataset[k]
    image = np.array(sample["image"].permute(1, 2, 0))
    img_h, img_w, c = image.shape
    axes[i].imshow(image)
    axes[i].imshow(sample["head_mask"][0], alpha=0.4)
    axes[i].imshow(
        cv2.resize(sample["gaze_heatmap"].numpy(), (img_w, img_h)), alpha=0.4
    )
    axes[i].set_title(f"in-out: {sample['inout']}")
    axes[i].axis("off")
plt.savefig("test-augs.png", dpi=150, bbox_inches="tight")


# %%
import cv2

plt.figure(figsize=(14, 14))
plt.imshow(np.array(sample["image"]))
plt.imshow(sample["head_mask"][0], alpha=0.4)
plt.imshow(cv2.resize(sample["gaze_heatmap"].numpy(), sample["image"].size), alpha=0.4)
plt.title("is inside: {}".format(sample["inout"]))

# %%
import matplotlib.pyplot as plt

from datasets import GazePredictionDataset
from src.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCropSafeGaze,
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

root = "demo"
file_name = "iDg2hR1FPgA_47999-48192.mp4"  # "3mXFJ8S-boU_4819.jpg"  #
annotations_file = "iDg2hR1FPgA_47999-48192.csv"  # "3mXFJ8S-boU_4819.csv"


transform = Compose(
    [
        Resize(img_size=(224, 224), head_size=(224, 224)),
        ToTensor(),
        Normalize(),
    ]
)
dataset = GazePredictionDataset(root, file_name, annotations_file, None)
sample = dataset[1]

# %%
plt.imshow(sample["image"])
plt.imshow(sample["head_mask"][0], alpha=0.5)

# %%

from src.visualize import draw_gaze

image = sample["image"]
gaze_point = [[0.871, 0.447], [0.77, 0.752], [0.787, 0.7417]]
inout = [0.7, 0.9, 1.0]
head_bbox = [[965, 49, 1213, 332], [386, 238, 671, 514], [1599, 373, 1839, 624]]
person_id = [0, 1, 2]
img_out = draw_gaze(image, gaze_point, inout, head_bbox, person_id)

# %%
plt.figure(figsize=(12, 8))
plt.imshow(img_out)
# %%
