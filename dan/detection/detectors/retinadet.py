import torch.nn as nn
from dan.design.builder import build_backbone, build_plugin, build_neck, DETECTORS
import torch
from torchvision.models import _utils
from dan.classifier.backbones import MobileBackboneV1   # remove
from dan.detection.necks import FPN, SSH


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anhors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anhors*2, (1, 1), 1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BoxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BoxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, (1, 1), 1, 0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, (1, 1), 1, 0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


@DETECTORS.register_module()
class RetinaDet(nn.Module):
    def __init__(self, backbone, return_layers, neck=None, plugin=None, train_cfg=None, test_cfg=None):
        # assert scene in ('face', 'ocr', 'general'), 'scene:{} is not support'.format(scene)
        super(RetinaDet, self).__init__()
        self.phase = train_cfg.phase
        self.scene = train_cfg.scene
        self.num_stages = 3  # 3, 4: small, mid, big
        self.anchor_num = 2   # diverse: face, general, word, different scene  
        self.return_layers = return_layers
        self.out_channels = 64
        self.backbone = build_backbone(backbone)
        
        self.body = _utils.IntermediateLayerGetter(self.backbone, self.return_layers)   # from a classifier 
        # self.body = self.backbone

        self.neck = build_neck(neck)
        
        self.ssh1 = build_plugin(plugin)
        self.ssh2 = build_plugin(plugin)
        self.ssh2 = build_plugin(plugin)
 
        self.BoxHead = self._make_bbox_head(fpn_num=self.num_stages, inchannels=self.out_channels, anchor_num=self.anchor_num)
        self.ClassHead = self._make_class_head(fpn_num=self.num_stages, inchannels=self.out_channels, anchor_num=self.anchor_num)
        self.LandmarkHead = None
        if self.scene == 'face':
            self.LandmarkHead = self._make_landmark_head(fpn_num=self.num_stages, inchannels=self.out_channels, anchor_num=self.anchor_num)

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=9):
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BoxHead(inchannels, anchor_num))
        return bboxhead

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=9):
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=9):
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        fpn = self.neck(out)
        
        # print(fpn[0].shape, fpn[1].shape, fpn[2].shape)
        # torch.Size([32, 64, 80, 80]) torch.Size([32, 64, 40, 40]) torch.Size([32, 64, 20, 20])
        
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh2(fpn[2])
        features = [feature1, feature2, feature3]
        

        box_regressions = torch.cat([self.BoxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        if self.LandmarkHead is not None:
            ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        else:
            ldm_regressions = None
        # print(box_regressions.shape, classifications.shape, ldm_regressions.shape, "xx")
        # torch.Size([32, 25200, 4]) torch.Size([32, 25200, 2]) torch.Size([32, 25200, 10])
        if self.phase == 'train':
            output = (box_regressions, classifications, ldm_regressions)
        else:
            output = (box_regressions, nn.functional.softmax(classifications, dim=-1), ldm_regressions)
        # torch.Size([32, 16800, 4]) torch.Size([32, 16800, 2]) torch.Size([32, 16800, 10])
        return output
