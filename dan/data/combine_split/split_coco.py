import os.path as osp

from pycocotools.coco import COCO
from dan.data.fileio import load, dump

from dan.data.utils.dictop import split_dict, sort_dict


class SplitDataset(object):
    """coco-like datasets analysis, at present, it's not work as expect"""
    def __init__(self, ann_file):
        self.ann_file = ann_file
        self.coco_dataset = load(ann_file)
        self.COCO = COCO(ann_file)
        # here our labelme2coco not care info(simple), but coco label original has
        self.info = self.coco_dataset['info'] if 'info' in self.coco_dataset else None

    def split_dataset(self, val_size=0.1, to_file=None):
        imgs_train, imgs_val = split_dict(self.COCO.imgs, val_size)
        print('images: {} train, {} test.'.format(len(imgs_train),
                                                  len(imgs_val)))

        # deal train data
        train = dict(info=self.info ,
                     categories=self.coco_dataset['categories'])
        train['images'] = list(imgs_train.values())  # bad design
        anns = []
        for key in imgs_train.keys():
            anns += self.COCO.imgToAnns[key]
            train['annotations'] = anns

        # deal test data
        val = dict(info=train['info'], categories=train['categories'])
        val['images'] = list(imgs_val.values())
        anns = []
        for key in imgs_val.keys():
            anns += self.COCO.imgToAnns[key]
            val['annotations'] = anns

        if to_file:
            path, name = osp.split(to_file)
            dump(train, file=osp.join(path, 'train_' + name), indent=4)
            dump(val, file=osp.join(path, 'val_' + name), indent=4)
        return val, train

    def split_cats(self):
        catToImgs = sort_dict(self.COCO.catToImgs)
        self.catToDatasets = []
        for cat, img_ids in catToImgs.items():
            img_ids = set(img_ids)
            categories = [
                cat_info for cat_info in self.coco_dataset['categories']
                if cat_info['id'] == cat
            ]
            images = [
                img_info for img_info in self.coco_dataset['images']
                if img_info['id'] in img_ids
            ]
            annotations = [
                ann_info for ann_info in self.coco_dataset['annotations']
                if ann_info['category_id'] == cat
            ]
            self.catToDatasets.append({
                'info': self.info,
                'categories': categories,
                'images': images,
                'annotations': annotations
            })
            
            
    def cats_children(self):
        pass

    def save_cat_datasets(self, to_file):
        for dataset in self.catToDatasets:
            dump(dataset, file=to_file.format(dataset['categories'][0]['name']), indent=4)
