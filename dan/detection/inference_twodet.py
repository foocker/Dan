from dan.data.coco_vison import get_data_loader, get_transform, collate_fn
from .detectors.fastercnn import fasterrcnn_resnetxx_fpnxx
import torch
import cv2
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX

cfg_two_stage = {
    'root': '/aidata/dataset/VOC/VOCdevkit/VOC2007/JPEGImages',
    'annFile':
    '/aidata/dataset/VOC/VOCdevkit/VOC2007/AnnotationsJson/voc2007_to_coco.json',
    'transform': get_transform(),
    'batch_size': 8,
    'collate_fn': collate_fn
}

Struct_Component_Cfg = {
    'backbone_pretrained': True,
    'backbone_name': 'resnet50',
    'fpn': {
        'return_layers': {
            'layer1': 0,
            'layer2': 1,
            'layer3': 2,
            'layer4': 3
        },
        'out_channels': 256
    },
    'num_classes': 21,  # (including the background).
    'anchor_generator': {
        'sizes': ((32, ), (64, ), (128, ), (256, ), (512, )),
        'aspect_ratios': ((0.5, 1.0, 2.0), ) * 5
    },
    'box_roi_pool': {
        'featmap_names': [0, 1, 2, 3],
        'output_size': 7,
        'sampling_ratio': 2
    },  # Consistent with fpn
    'fasterrcnn_pretrained': ''  # weight path
}

PRE_DEFINE_CATEGORIES = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

CATEGORIES_INVERSE = {v: k for k, v in PRE_DEFINE_CATEGORIES.items()}


def sparse_detections(detections):
    boxes, labels, scores = detections['boxes'], detections[
        'labels'], detections['scores']
    index_selected = scores > 0.1
    boxes = boxes[index_selected].cpu().numpy()
    labels = labels[index_selected].cpu().numpy()
    scores = scores[index_selected].cpu().numpy()
    return boxes, labels, scores


def visualiz_inference(img, boxes, labels, scores):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box, id_, score in zip(boxes, labels, scores):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        cv2.putText(img,
                    text='{}_{:.3f}'.format(CATEGORIES_INVERSE[id_], score),
                    org=(x1 + 5, y1 + 5),
                    fontFace=font,
                    fontScale=1,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                    color=(0, 255, 0))
    plt.imshow(img)
    # plt.show()
    return img
    


def temp_test():
    model = fasterrcnn_resnetxx_fpnxx(Struct_Component_Cfg)
    model = model.to('cuda:0')
    model.eval()
    weights = '/vdata/Synthesize/weights_detect/datanamevoc07_to_coco_epoch_129_loss_0.0433.pth'
    # weights = '/vdata/Synthesize/weights_detect/resnet18_voc07_to_coco_epoch_99_loss_0.7557.pth'
    state_dict = torch.load(weights)
    print(state_dict.keys())
    print(state_dict['model'].keys())
    model.load_state_dict(state_dict['model'])

    data_loader = get_data_loader(**cfg_two_stage)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(data_loader):
            imgs = list(img.to('cuda:0') for img in imgs)
            detections = model(imgs)
            # print(i, detections)
            # if i == 1:
            #     break
            for img, detection in zip(imgs, detections):
                img = img.cpu().permute(1, 2, 0).numpy()
                boxes, labels, scores = sparse_detections(detection)
                print(labels)
                _ = visualiz_inference(img, boxes, labels, scores)
            if i == 0:
                break
