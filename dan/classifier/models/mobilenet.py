# from ..backbones import MobileBackboneV1, MobileBackboneV2

# from torchvision.models.utils import load_state_dict_from_url

from torch import nn
from dan.design.builder import build_backbone,build_head, CLASSIFIER


# model_urls = {
#     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
# }

# def mobilenet_v2(pretrained=False, progress=True, **kwargs):
#     """
#     Constructs a MobileNetV2 architecture from
#     `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = MobileNetV2(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model


@CLASSIFIER.register_module()
class MobilenetV1(nn.Module):
    def __init__(self, backbone, head=None, **kwargs):
        super(MobilenetV1, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


@CLASSIFIER.register_module()
class MobileNetV2(nn.Module):
    def __init__(self, backbone, head=None, **kwargs):
        super(MobileNetV2, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x