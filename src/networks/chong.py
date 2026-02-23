import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.utils import generate_gaze_map, spherical2cartesian

# ==================================================================================================================== #
#                                                   CHONG ARCHITECTURE                                                 #
# ==================================================================================================================== #


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ChongNet(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(
        self,
        block=Bottleneck,
        layers_scene=[3, 4, 6, 3, 2],
        layers_face=[3, 4, 6, 3, 2],
    ):
        # Resnet Feature Extractor
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super().__init__()

        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # scene pathway
        self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1)  # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1)  # additional to resnet50

        # attention
        self.attn = nn.Linear(1808, 1 * 7 * 7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes_scene,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes_face,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)

    def forward(self, images, head, face):
        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2)  # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((images, head), dim=1)
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        # attn_weights = torch.ones(attn_weights.shape)/49.0
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat)  # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)

        # scene + face feat -> in/out
        encoding_inout = self.compress_conv1_inout(scene_face_feat)
        encoding_inout = self.compress_bn1_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = self.compress_conv2_inout(encoding_inout)
        encoding_inout = self.compress_bn2_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = encoding_inout.view(-1, 49)
        encoding_inout = self.fc_inout(encoding_inout)

        # scene + face feat -> encoding -> decoding
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x, torch.mean(attn_weights, 1, keepdim=True), encoding_inout

# ==================================================================================================================== #
#                                                 BASELINE ARCHITECTURE                                                #
# ==================================================================================================================== #
def normalize_map(gaze_maps):
    bs, _ = gaze_maps.shape
    min_val, _ = gaze_maps.min(dim=1)
    max_val, _ = gaze_maps.max(dim=1)
    gaze_prob = (gaze_maps - min_val.view(-1, 1)) / (max_val.view(-1, 1) - min_val.view(-1, 1))
    return gaze_prob


class GazeBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        self.gaze360 = GazeStatic()
        # ckpt_path = "/idiap/temp/stafasca/projects/rinnegan/weights/gaze360_resnet18.pt"
        # checkpoint = torch.load(ckpt_path, map_location="cuda")
        # self.gaze360.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.gaze360.eval()
        # print("Loaded gaze360 weights from {}".format(ckpt_path))

        self.adapter = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)
        backbone = models.resnet50(weights="IMAGENET1K_V2")
        self.encoder = nn.ModuleList(backbone.children())[:-2]
        self.decoder = Decoder(in_channels=2048)

        # Initialize weights of adapter
        n = self.adapter.kernel_size[0] * self.adapter.kernel_size[1] * self.adapter.out_channels
        self.adapter.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, sample):
        image = sample["image"]

        # ============================================================================================ #
        # head_mask = sample["head_masks"].squeeze(1)  # (B, 1, 1, H, W) >> (B, 1, H, W)
        # x = torch.cat([image, head_mask], dim=1)
        # x = torch.relu(self.adapter(x))
        # ============================================================================================ #

        # ============================================================================================ #
        # x = image
        # ============================================================================================ #

        # ============================================================================================ #
        # Predict gaze map
        device = sample["image"].device
        bs = len(sample["image"])
        head = sample["heads"].squeeze(1)
        left_eye = torch.zeros((bs, 3, 64, 64), device=device)
        right_eye = torch.zeros((bs, 3, 64, 64), device=device)

        with torch.no_grad():
            # Predict 3D gaze
            gaze2d, offset = self.gaze360(head, left_eye, right_eye)
            # Build gaze map
            gaze_vecs = -spherical2cartesian(gaze2d)
            coords = (sample["head_centers"].squeeze(1) * 224).int()  # TODO: 224 needs to be parametrized
            bbox_center_depth = sample["depth"].squeeze(1)[torch.arange(bs), coords[:, 1], coords[:, 0]]
            origin_pts = torch.cat([sample["head_centers"].squeeze(1), bbox_center_depth.view(-1, 1)], dim=1)
            gaze_map = generate_gaze_map(gaze_vecs, origin_pts, sample["depth"], size=(224, 224), transform=normalize_map)
            # Fill head box with zeros
            gaze_map = gaze_map * (1 - sample["head_masks"].squeeze())

        x = torch.cat([image, gaze_map.unsqueeze(1)], dim=1)
        x = torch.relu(self.adapter(x))
        # ============================================================================================ #

        # Encode
        for k in range(4):
            x = self.encoder[k](x)

        # Hooks
        z1 = self.encoder[4](x)  # (N, 256, 56, 56)
        z2 = self.encoder[5](z1)  # (N, 512, 28, 28)
        z3 = self.encoder[6](z2)  # (N, 1024, 14, 14)
        x = self.encoder[7](z3)  # (N, 2048, 7, 7)

        # Decode
        x = self.decoder(x, [z1, z2, z3])

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=0)
        self.bn11 = nn.BatchNorm2d(in_channels // 2)
        self.conv1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(in_channels // 2)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 2**2, kernel_size=3, stride=2, padding=0)
        self.bn21 = nn.BatchNorm2d(in_channels // 2**2)
        self.conv2 = nn.Conv2d(in_channels // 2**2, in_channels // 2**2, kernel_size=3, stride=1, padding=1)
        self.bn22 = nn.BatchNorm2d(in_channels // 2**2)

        self.deconv3 = nn.ConvTranspose2d(in_channels // 2**2, in_channels // 2**3, kernel_size=4, stride=2, padding=0)
        self.bn31 = nn.BatchNorm2d(in_channels // 2**3)
        self.conv3 = nn.Conv2d(in_channels // 2**3, in_channels // 2**3, kernel_size=3, stride=1, padding=1)
        self.bn32 = nn.BatchNorm2d(in_channels // 2**3)

        self.conv4 = nn.Conv2d(in_channels // 2**3, 1, kernel_size=1, stride=1, padding=0)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, zs):
        z1, z2, z3 = zs

        x = torch.relu(self.bn11(self.deconv1(x)))  # (N, 1024, 15, 15)
        x = x + F.pad(z3, (1, 0, 0, 1), "constant", 0)  # (N, 1024, 15, 15)
        x = torch.relu(self.bn12(self.conv1(x)))  # (N, 1024, 15, 15)

        x = torch.relu(self.bn21(self.deconv2(x)))  # (N, 512, 31, 31)
        x = x + F.pad(z2, (2, 1, 1, 2), "constant", 0)  # (N, 512, 31, 31)
        x = torch.relu(self.bn22(self.conv2(x)))  # (N, 512, 31, 31)

        x = torch.relu(self.bn31(self.deconv3(x)))  # (N, 256, 64, 64)
        x = x + F.pad(z1, (4, 4, 4, 4), "constant", 0)  # (N, 256, 64, 64)
        x = torch.relu(self.bn32(self.conv3(x)))  # (N, 256, 64, 64)

        x = self.conv4(x)  # (N, 1, 64, 64)

        return x


class GazeStatic(nn.Module):
    def __init__(self):
        super(GazeStatic, self).__init__()

        self.feature_dim = 512  # the dimension of the CNN feature to represent each frame
        # Build Network Base
        self.base_head = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_head = nn.Sequential(*list(self.base_head.children())[:-1])
        # Build Network Base Eye
        self.base_eye = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_eye = nn.Sequential(*list(self.base_eye.children())[:-1])

        # Build Network Head
        dummy_head = torch.empty((1, 3, 224, 224))
        dummy_head = self.base_head(dummy_head)
        dummy_eye = torch.empty((1, 3, 64, 64))
        dummy_eye = self.base_eye(dummy_eye)
        self.head = nn.Sequential(
            nn.Linear(dummy_head.size(1) + 2 * dummy_eye.size(1), self.feature_dim), nn.ReLU(inplace=True), nn.Linear(self.feature_dim, 3)
        )  # 3 = number of outputs (yaw, pitch, offset)

    def forward(self, head, left_eye, right_eye):
        # Model output
        h = self.base_head(head)  # Nx512x1x1
        le = self.base_eye(left_eye)  # Nx512x1x1
        re = self.base_eye(right_eye)  # Nx512x1x1

        x = torch.concat([h.view(h.size(0), -1), le.view(le.size(0), -1), re.view(re.size(0), -1)], dim=1)  # Nx1536
        out = self.head(x)

        # Compute spherical gaze
        yaw = math.pi * torch.tanh(out[:, 0]).view(-1, 1)
        pitch = math.pi * 1 / 2 * torch.tanh(out[:, 1]).view(-1, 1)
        gaze2d = torch.hstack([yaw, pitch])
        # Compute offset
        offset = math.pi * torch.sigmoid(out[:, 2])
        offset = offset.view(-1, 1).expand(offset.size(0), 2)

        return gaze2d, offset
