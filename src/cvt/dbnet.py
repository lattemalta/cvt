import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, layers: List[int] = [2, 2, 2, 2]):
        super().__init__()
        self.in_planes = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        
    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # Extract multi-scale features
        c2 = self.layer1(x)   # 1/4 scale
        c3 = self.layer2(c2)  # 1/8 scale
        c4 = self.layer3(c3)  # 1/16 scale
        c5 = self.layer4(c4)  # 1/32 scale
        
        return {
            'c2': c2,  # 64 channels
            'c3': c3,  # 128 channels
            'c4': c4,  # 256 channels
            'c5': c5   # 512 channels
        }


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channel, out_channels, kernel_size=1, bias=False)
            for in_channel in in_channels
        ])
        
        # Output convolutions (3x3 conv for smoothing)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for _ in in_channels
        ])
        
        # Batch normalization layers
        self.lateral_bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in in_channels
        ])
        
        self.fpn_bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in in_channels
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Build lateral connections
        laterals = []
        for feature, lateral_conv, lateral_bn in zip(features, self.lateral_convs, self.lateral_bns):
            lateral = F.relu(lateral_bn(lateral_conv(feature)), inplace=True)
            laterals.append(lateral)
        
        # Build top-down path
        # Start from the highest level feature
        fpn_features = [laterals[-1]]
        
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample the higher-level feature
            upsampled = F.interpolate(
                fpn_features[-1], 
                size=laterals[i].shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # Add lateral connection
            fused = laterals[i] + upsampled
            fpn_features.append(fused)
        
        # Reverse to get bottom-up order
        fpn_features = fpn_features[::-1]
        
        # Apply final convolution for smoothing
        outputs = []
        for feature, fpn_conv, fpn_bn in zip(fpn_features, self.fpn_convs, self.fpn_bns):
            output = F.relu(fpn_bn(fpn_conv(feature)), inplace=True)
            outputs.append(output)
            
        return outputs


class AdaptiveScaleFusion(nn.Module):
    """Adaptive Scale Fusion module for DBNet++"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        # Stage-wise attention (which scale/level is important?)
        self.stage_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention (which spatial location is important?)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Resize all features to the same size (largest feature map size)
        target_size = features[0].shape[2:]  # Use P2 size (highest resolution)
        resized_features = []
        
        for feature in features:
            if feature.shape[2:] != target_size:
                resized = F.interpolate(
                    feature, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                resized = feature
            resized_features.append(resized)
        
        # Concatenate features along channel dimension
        concat_features = torch.cat(resized_features, dim=1)
        
        # Apply stage-wise attention (channel attention)
        stage_att_weights = self.stage_attention(concat_features)
        stage_weighted = concat_features * stage_att_weights
        
        # Apply spatial attention
        spatial_att_weights = self.spatial_attention(stage_weighted)
        final_features = stage_weighted * spatial_att_weights
        
        return final_features


class DBHead(nn.Module):
    """Differentiable Binarization Head for DBNet++"""
    
    def __init__(self, in_channels: int, k: int = 50):
        super().__init__()
        self.k = k  # Scale factor for differentiable binarization
        
        # Probability map head (P)
        self.prob_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
        # Threshold map head (T)
        self.thresh_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Generate probability map
        prob_map = self.prob_head(x)
        
        # Generate threshold map
        thresh_map = self.thresh_head(x)
        
        # Differentiable binarization
        if self.training:
            # During training, use differentiable approximation
            binary_map = 1.0 / (1.0 + torch.exp(-self.k * (prob_map - thresh_map)))
        else:
            # During inference, use step function
            binary_map = (prob_map >= thresh_map).float()
        
        return {
            'prob_map': prob_map,
            'thresh_map': thresh_map, 
            'binary_map': binary_map
        }


class DBNet(nn.Module):
    """Complete DBNet++ model for text detection"""
    
    def __init__(self, backbone_type: str = 'resnet18'):
        super().__init__()
        
        # Backbone network
        if backbone_type == 'resnet18':
            self.backbone = resnet18_backbone()
            backbone_channels = [64, 128, 256, 512]
        elif backbone_type == 'resnet34':
            self.backbone = resnet34_backbone()
            backbone_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(backbone_channels, out_channels=256)
        
        # Adaptive Scale Fusion (DBNet++ enhancement)
        self.asf = AdaptiveScaleFusion(in_channels=256 * 4)  # 4 scales × 256 channels
        
        # DB Head for text detection
        self.db_head = DBHead(in_channels=256 * 4)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features from backbone
        backbone_features = self.backbone(x)
        feature_list = [backbone_features['c2'], backbone_features['c3'], 
                       backbone_features['c4'], backbone_features['c5']]
        
        # Feature Pyramid Network
        fpn_features = self.fpn(feature_list)
        
        # Adaptive Scale Fusion
        fused_features = self.asf(fpn_features)
        
        # DB Head
        outputs = self.db_head(fused_features)
        
        return outputs


def resnet18_backbone() -> ResNetBackbone:
    return ResNetBackbone([2, 2, 2, 2])


def resnet34_backbone() -> ResNetBackbone:
    return ResNetBackbone([3, 4, 6, 3])