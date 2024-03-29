import os
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .transforms import get_augumentation


class CocoDataset(Dataset):
    # target is 4+1
    def __init__(self, root_dir, set_name='train2017', transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.coco = COCO(
            os.path.join(self.root_dir, 'annotations',
                         'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_image(self, image_idx):
        image_info = self.coco.loadImgs(self.image_ids[image_idx])[0]

        # path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        path = os.path.join(self.root_dir, 'images', image_info['file_name'])

        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def load_annotations(self, image_idx):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_idx],
                                              iscrowd=False)
        annotations = np.zeros((0, 5))
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] , coco format to pascal_voc format [x1, y1, x2, y2], beacuse MultiBoxLoss, wider face data
        # when ablu coco format, remove below
        # annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        # annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_idx):
        image = self.coco.loadImgs(self.image_ids[image_idx])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        h, w, _ = img.shape
        annot = self.load_annotations(idx)
        targets = np.array(annot)
        bboxes = targets[:, :
                         4]  # h == w default /h  for albumentations box format
        labels = targets[:, 4]
        labels = np.array(labels, dtype=np.int)
        if self.transform is not None:
            annotation = {
                'image': img,
                'bboxes': bboxes,
                'category_id': labels
            }
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bboxes = augmentation['bboxes']
            labels = augmentation['category_id']

        return {'image': img, 'bboxes': bboxes, 'category_id': labels}

    def __len__(self):
        return len(self.image_ids)


def coco_collate_primary(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5)) * -1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                annot_padded[idx, :len(annot), 4] = lab
    return torch.stack(imgs, 0), torch.FloatTensor(annot_padded)


def coco_collate(batch):
    imgs = [s['image'] for s in batch
            ]  # tensor, h, w, c->c, h, w , handle at transform in __getitem__
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5)) * -1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                #  xywh-->x1,y1,x2,y2 for general box,ssd target assigner
                annot_padded[idx, :len(annot),
                             2] += annot_padded[idx, :len(annot), 0]
                #contains padded -1 label
                annot_padded[idx, :len(annot),
                             3] += annot_padded[idx, :len(annot), 1]
                # priorbox for ssd primary target assinger
                annot_padded[idx, :len(annot), :] /= 640
                annot_padded[idx, :len(annot), 4] = lab
    return torch.stack(imgs, 0), torch.FloatTensor(annot_padded)


# def coco_collate(batch):
#     imgs = [s['image'] for s in batch]    # tensor, h, w, c->c, h, w , handle at transform in __getitem__
#     annots = [s['bboxes'] for s in batch]
#     labels = [s['category_id'] for s in batch]

#     targets = []
#     imgs = []
#     for _, sample in enumerate(batch):
#         for _, img_anno in enumerate(sample):
#             if torch.is_tensor(img_anno):
#                 imgs.append(img_anno)
#             elif isinstance(img_anno, np.ndarray):
#                 annos = torch.from_numpy(img_anno).float()
#                 targets.append(annos)
#     return torch.stack(imgs, 0), targets


def test_coco_aug():
    import matplotlib
    import random
    from torch.utils import data
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data_basedir = '/aidata/dataset/cigarette/cig_mask_coco'
    # tp = None
    tp = get_augumentation(phase='train', width=640, height=640, ft='coco')
    dataset = CocoDataset(data_basedir, set_name='annotations', transform=tp)
    # dataset = CocoDataset(data_basedir, set_name='annotations', transform=tp)
    num = dataset.__len__()
    image_idx = random.randint(0, num)
    print(image_idx)
    print('having {} images'.format(num))
    dataone = dataset.__getitem__(image_idx)
    img, bboxes, catg = dataone['image'], dataone['bboxes'], dataone[
        'category_id']

    # -----test coco_collate for ablu xywh--> xyxy------
    epoch_iterator = iter(
        data.DataLoader(dataset,
                        1,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=coco_collate))
    imgs, targets = next(epoch_iterator)
    print('bf', imgs.shape, targets.shape)
    if tp is not None:
        # bs = 1
        imgs = imgs.squeeze(0)
        imgs *= 255
        imgs = imgs.permute(1, 2, 0).cpu().numpy()
        targets = targets.squeeze(0)[:, :4].cpu().numpy()
    print('af', imgs.shape, targets.shape, targets)
    for box in targets:
        # coco format, after tp, keep x, y, w, h
        l, t, l_right, t_down = map(int, box * 640)
        # ssd format, after tp, keep x, y, w, h
        # l, t, l_w, t_h = map(int, box*640)
        # print(l, t, l_right, t_down, imgs.shape, type(imgs))
        img_rec = cv2.rectangle(imgs, (l, t), (l_right, t_down), (0, 0, 255),
                                2)
        # img_rec = cv2.rectangle(imgs, (l, t), (l + l_w, t + t_h), (0, 0, 255), 2)

    cv2.imwrite('rectangle_tp_point_collate_priobox_{}.jpg'.format(image_idx),
                img_rec)
    # -----test coco_collate for ablu xywh--> xyxy------

    # ------ test for CocoDataset------
    # if tp is not None:
    #     img *= 255
    #     img = img.permute(1, 2, 0).numpy()
    #     # img = img.numpy().transpose(1, 2, 0)
    # print(len(dataset.classes), dataset.num_classes())
    # print('box', bboxes, 'label', catg)
    # for box in bboxes:
    #     l, t, l_right, t_down = map(int, box)    # coco format, after tp, keep x, y, w, h
    #     print(l, t, l_right, t_down, img.shape, type(img))

    #     img_rec = cv2.rectangle(img, (l, t), (l_right+l, t_down+t), (0, 0, 255), 2)
    #     #
    # image_info = dataset.coco.loadImgs(dataset.image_ids[image_idx])[0]
    # print(image_info)
    # cv2.imwrite('rectangle_tp_point_' + image_info['file_name'].split('/')[-1], img_rec)
    # img_rec = cv2.rectangle(img, (l, t), (l_right, t_down), (0, 0, 255), 2)
    # ------ test for CocoDataset------
