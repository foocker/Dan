from dan.design import Config, Registry
from dan.detection.detectors import RetinaDet
from dan.design.builder import build_plugin, build_backbone, build_detector
# from dan.detection.netblocks import FPN


# cfg = Config.fromfile("dan/detection/config/face_config.py")

# print(cfg.__dir__())
# print(cfg._cfg_dict)
# print(cfg.model,"\n", cfg.train_cfg)
# print(cfg.train_cfg.cfg_detct.scales, "\n", cfg.train_cfg.xx.save_weights)
# print(cfg.test_cfg)

# det = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

# plugin = build_plugin(cfg.model.plugin)

# backb = build_backbone(cfg.model.neck)

# print(det)
# print(plugin)
# print(backb)

# reg = Registry("HEATMAP")
# print(reg.module_dict)

from torchvision.models import resnet18

model = resnet18()


def tag2layer(net, tags, parameter_mode='.weight'):
    '''
    parameter_mode='.weight', '.bias', '.grad'
    may diff net, the rule is diff, so todo...
    '''
    know_number = {str(i):'[{}]'.format(i) for i in range(10)}
    
    tag_split = tags.split('.')
    print(tag_split)
    call_str = ''.join([know_number[l_n] if l_n in know_number else '.'+l_n  for l_n in tag_split])
    if parameter_mode != '.grad':
        call_str = net + call_str + parameter_mode
    else:
        call_str = net + call_str + '.weight' + '.grad'
    
    print(call_str)
        
    return eval(call_str)

h = tag2layer('model', 'layer2.1.conv2', '.weight')  # layer4.1.relu has no weight, BatchNorm2d' object has no attribute 'grad'
print(h.shape)
        
# for name, layer in model.named_modules():
#     print(name, type(name))
#     if name == 'conv1':
#         strings = 'model' + '.' + name + '.weight'
#         x = eval('model' + '.' + name + '.weight')
#         # print(x.shape)
#         print(model.conv1.weight.shape)
        

# print(eval('2+3'))
# print(eval('model'))

# layer1.0.conv2
# print(model.layer1[0].conv2.weight)
