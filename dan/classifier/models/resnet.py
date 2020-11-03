from torch import nn
from ..backbones import ResBackbone
from dan.design.builder import CLASSIFIER, build_backbone, build_head


@CLASSIFIER.register_module()
class ResNet(nn.Module):
    def __init__(self, cfg, backbone, head=None, **kwargs):
        super(ResNet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
        