from dan.design import Config, Registry
from dan.detection.detectors import RetinaDet
from dan.detection.builder import build_plugin, build_backbone, build_detector
# from dan.detection.netblocks import FPN



cfg = Config.fromfile("dan/detection/config/face_config.py")

# print(cfg.__dir__())
# print(cfg._cfg_dict)
# print(cfg.model,"\n", cfg.train_cfg)
# print(cfg.train_cfg.cfg_detct.scales, "\n", cfg.train_cfg.xx.save_weights)
# print(cfg.test_cfg)

det = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

# plugin = build_plugin(cfg.model.plugin)

# backb = build_backbone(cfg.model.neck)

print(det)
# print(plugin)
# print(backb)

# reg = Registry("HEATMAP")
# print(reg.module_dict)