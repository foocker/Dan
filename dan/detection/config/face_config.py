model = dict(
    type='RetinaDet',
    backbone=dict(
        type='MobileBackboneV1',
    ),
    # return_layers={'stage1': 1, 'stage2': 2, 'stage3': 3},
    neck=dict(
        type='FPN',
        in_channels_list=[32*2, 32*4, 32*8],
        out_channels=64
    ),
    plugin=dict(
        type='SSH',
        in_channel=64,
        out_channel=64)
    )

train_cfg = dict(
    cfg_detct={
    'anchor_sizes': '',
    'aspect_ratios': '',
    'anchor_strides': '',
    'straddle_thresh': '',
    'octave': '',
    'scales_per_octave': 3,
    'ratios': [0.5, 1, 2],
    'scales': [2**0, 2**(1.0/3), 2**(2.0/3)],
    'pyramid_levels': [3, 4, 5]
    },
    phase='train',
    scene='general',
    xx = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    
    'scales_per_octave': 3,
    'ratios': [0.5, 1, 2],
    'scales': [2 ** 0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)],
    'pyramid_levels': [3, 4, 5],
    
    # train
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 500,
    'decay1': 400,
    'decay2': 450,
    'image_size': 640,
    'pretrain': False,
    'rgb_means': (104, 117, 123),
    'num_classes': 2,
    'num_workers': 4,
    'momentum': 0.9,
    'lr': 1e-3,
    'gamma': 0.1,
    'weight_decay': 5e-5,
    'resume_epoch': 0,
    # 'training_dataset': '/aidata/dataset/xcb_dataset/xcb_manage/detect',
    # 'training_label': '/aidata/dataset/xcb_dataset/xcb_manage/detect/annotations/annotations_train.json',
    'training_dataset': '/aidata/dataset/cigarette/cig_mask_coco',
    'training_label': '/aidata/dataset/cigarette/cig_mask_coco/annotations.json',
    'save_weights': '/vdata/Synthesize/weights_detect/',
    'weights_label': '_cigarette_box',
    }
)

test_cfg= dict(
    
)