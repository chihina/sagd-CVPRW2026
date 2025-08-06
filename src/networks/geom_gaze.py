import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import pickle

from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms import Resize
import torch.nn.functional as F


# returns 3DFoV; resnet/efficientnet + prediction head
class HumanCentric(nn.Module):
    def __init__(self):
        super(HumanCentric, self).__init__()
        
        self.feature_dim = 512  # the dimension of the CNN feature to represent each frame
        # Build Network Base
        base = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        
        # Build Network Head
        dummy_head = torch.empty((1, 3, 224, 224))
        dummy_head = self.backbone(dummy_head)            
        self.head_new = nn.Sequential(
                        nn.Linear(dummy_head.size(1), self.feature_dim), 
                        nn.ReLU(inplace=True),
                        nn.Linear(self.feature_dim, 3),    # predict 3D unit vector in eye coordinate space
                        nn.Tanh()) 
        
        self.fov_const = np.exp(4.5) / 0.9

    def forward(self, head, pcd, cam2eye, head_centers):
        # Model output
        h = self.backbone(head).squeeze(dim=-1).squeeze(dim=-1) # Nx512   
        head_embedding = h.clone()
        
        direction = self.head_new(h) 
        # convert to unit vector
        normalized_direction = direction / direction.norm(dim=1, keepdim=True)
        normalized_direction = normalized_direction.unsqueeze(2)
        # convert to camera coordinate system
        normalized_direction_cam = torch.linalg.solve(cam2eye, normalized_direction)
            
        # normalize point cloud
        pcd = pcd.permute(0,2,3,1)
        head_centers_3d = pcd[range(pcd.shape[0]), head_centers[:, 1], head_centers[:, 0]]
        padding_indices = pcd.sum(-1)==0
        pcd -= head_centers_3d.unsqueeze(1).unsqueeze(1)
        pcd = F.normalize(pcd, p=2, dim=-1)
        
        # get 3DFoV
        batch_size, height, width, _ = pcd.shape
        pcd = pcd.view(batch_size, height*width, 3)        
        similarity_map = torch.matmul(pcd, normalized_direction_cam)
        
        fov = similarity_map.view(batch_size, height, width)
        fov = (fov * (fov > 0.9)) + ((torch.exp(fov*5) / self.fov_const) * (fov <= 0.9))   # apply threshold on similarity map
        fov[padding_indices] = 0
    
        return fov.unsqueeze(1), normalized_direction.squeeze(-1), head_embedding


# efficientnet followed by an FPN
class FeatureExtractor(nn.Module):
    
    def __init__(self, backbone_name):
        
        '''
        args:
        backbone_name: name of the backbone to be used; ex. 'efficientnet-b0'
        '''
        
        super(FeatureExtractor, self).__init__()
    
        self.backbone = EfficientNet.from_pretrained(backbone_name)
        if backbone_name=='efficientnet-b3':
            self.fpn = FeaturePyramidNetwork([32, 48, 136, 384], 64)
        elif backbone_name=='efficientnet-b2':
            self.fpn = FeaturePyramidNetwork([24, 48, 120, 352], 64)
        elif backbone_name=='efficientnet-b0' or backbone_name=='efficientnet-b1':
            self.fpn = FeaturePyramidNetwork([24, 40, 112, 320], 64)        
        
    def forward(self, x):
        
        features = self.backbone.extract_endpoints(x)
        
        # select features to use
        fpn_features = OrderedDict()
        fpn_features['reduction_2'] = features['reduction_2']
        fpn_features['reduction_3'] = features['reduction_3']
        fpn_features['reduction_4'] = features['reduction_4']
        fpn_features['reduction_5'] = features['reduction_5']
        
        # upsample features from efficientnet using an FPN to generate features at (H/4, W/4) resolution
        features = self.fpn(fpn_features)['reduction_2']
        
        return features


# simple prediction head that takes the features and gaze cone to regress the attention heatmap
class PredictionHead(nn.Module):
    
    # output_size -> (width, height)
    def __init__(self, inchannels, output_size=(64,64)):
        super(PredictionHead, self).__init__()
        
        self.act = nn.ReLU()
        self.output_size = output_size
        
        self.conv1 = nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3)
        self.bn1 = nn.BatchNorm2d(inchannels)
        self.conv2 = nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3)
        self.bn2 = nn.BatchNorm2d(inchannels)
        self.conv3 = nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(inchannels)
        self.conv4 = nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3)
        self.bn4 = nn.BatchNorm2d(inchannels)
        self.conv5 = nn.Conv2d(inchannels, inchannels//2, 3, padding=3, dilation=3)
        self.bn5 = nn.BatchNorm2d(inchannels//2)
        self.conv6 = nn.Conv2d(inchannels//2, inchannels//4, 3, padding=3, dilation=3)
        self.bn6 = nn.BatchNorm2d(inchannels//4)
        self.conv7 = nn.Conv2d(inchannels//4, 1, 1)

    def forward(self, x):
        
        # upsample the features to output size
        if self.output_size==None:  # No upsampling
            output_size = (x.shape[3], x.shape[2])
        else:
            output_size = self.output_size
        x = nn.Upsample(size=(output_size[1], output_size[0]), mode='bilinear', align_corners=False)(x)
        x = self.act(self.bn1(self.conv1(x)))
        
        # regress the heatmap
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        x = self.act(self.bn6(self.conv6(x)))
        x = self.conv7(x)
        
        return x
        
        
# compress modality spatially
class CompressModality(nn.Module):
    
    def __init__(self, in_channels):
        super(CompressModality, self).__init__()
        
        self.act = nn.GELU()
        
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = nn.MaxPool2d(x.shape[2:])(x)

        return x.squeeze(dim=-1).squeeze(dim=-1)
    
    
# predicts in vs out gaze; CompressModality + Linear
class InvsOut(nn.Module):
    
    def __init__(self, in_channels):
        
        '''
        args:
        in_channels: number of input channels
        '''
        
        super(InvsOut, self).__init__()
        self.compress_inout = CompressModality(in_channels)
        self.inout = nn.Sequential(nn.Linear(1024, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
    
    def forward(self, x, head_embedding):
        
        x = self.compress_inout(x)
        x = torch.cat([x, head_embedding], axis=1)
        x = self.inout(x)
        
        return x
    

# Processes the image and predicted 3DFoV to predict an attention heatmap
class GeomGaze(nn.Module):
    
    def __init__(self, backbone_name='efficientnet-b1', output_size=(64,64)):
        
        '''
        args:
        backbone_name: name of the backbone to be used; ex. 'efficientnet-b0'
        output_size: size of predicted gaze heatmap
        '''
        
        super(GeomGaze, self).__init__()
        self.feature_extractor = FeatureExtractor(backbone_name)    
        self.prediction_head = PredictionHead(64, output_size)
        self.gaze_encoder = HumanCentric()
        self.in_vs_out_head = InvsOut(64)

        # add additional channels
        input_layer = self.feature_extractor.backbone._conv_stem.weight
        self.feature_extractor.backbone._conv_stem.weight = torch.nn.Parameter(torch.cat([input_layer.clone(), input_layer.clone()[:,:2,:,:]], axis=1))

    def forward(self, sample):
        batch_size, num_people, _, head_h, head_w = sample['heads'].squeeze(1).shape
        batch_size, _, img_h, img_w = sample['image'].squeeze(1).shape
        num_pairs = num_people*(num_people-1)
        device = sample['heads'].device
        
        # repeat image and pcd by number of people
        img = sample['image']
        img = img.repeat(1,num_people,1,1,1)
        img = img.view(-1, 3, img_h, img_w)
        pcd = sample['pcd']
        pcd = pcd.repeat(1,num_people,1,1,1)
        pcd = pcd.view(-1, 3, img_h, img_w)
        
        # get 3DFoV
        head = sample['heads'].view(-1, 3,head_h,head_w)
        cam2eye = sample['cam2eye'].view(-1,3,3)
        head_center = sample['head_centers'].view(-1, 2)
        head_center = (head_center* torch.tensor([img_w, img_h]).to(device)).int()
        head_mask = sample['head_masks'].view(-1,1,img_h,img_w)
        fov, gaze_vec_3d, head_embedding = self.gaze_encoder(head, pcd, cam2eye, head_center)
        
        # extract the features
        x = torch.cat([img, fov, head_mask], dim=1)    # concat image with 3DFoV and head location mask
        x = self.feature_extractor(x)
        
        # apply the prediction head to get the heatmap
        gaze_hm = self.prediction_head(x)
        output_size = gaze_hm.shape[-2:]
        
        # apply the in vs out head
        inout = self.in_vs_out_head(x, head_embedding)
#         inout = torch.zeros(batch_size, 1, num_people).to(device)
        
        # add extra dimension for time
        return gaze_vec_3d.view(batch_size, 1, num_people, -1), gaze_hm.view(batch_size, 1, num_people, output_size[0], output_size[1]), inout.view(batch_size, 1, num_people), torch.zeros(batch_size, 1, num_pairs).to(device), torch.zeros(batch_size, 1, num_pairs).to(device), torch.zeros(batch_size, 1, num_pairs).to(device)