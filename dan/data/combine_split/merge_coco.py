import copy
from dan.data.fileio import load, dump


class MergeCOCO(object):
    """merge multiple coco-like datasets into one file

    Args:
        files (list): a list of str or COCO object
    """
    def __init__(self, files):
        if not isinstance(files, (list, tuple)):
            raise TypeError('files must be a list, but got {}'.format(
                type(files)))
        assert len(files) > 1, 'least 2 files must be provided!'
        self.files = files
        if isinstance(self.files[0], dict):
            self.merge_coco = copy.deepcopy(self.files[0])
        else:
            self.merge_coco = load(self.files[0])
        self.img_ids = [
            img_info['id'] for img_info in self.merge_coco['images']
        ]
        self.ann_ids = [
            img_info['id'] for img_info in self.merge_coco['annotations']
        ]
        self.cat_ids = [
            img_info['id'] for img_info in self.merge_coco['categories']
        ]

    def update_img_ann_ids(self, images, anns, cats):
        img_id_map = dict()
        img_max_id = max(self.img_ids)
        for i in range(len(images)):
            img_id_map[images[i]['id']] = \
                images[i]['id'] + img_max_id + 1
            images[i]['id'] += img_max_id + 1
        self.merge_coco['images'] += images

        ann_max_id = max(self.ann_ids)
        ann_max_id_category = max(self.cat_ids)
        
        for c in cats:
            c['id'] += ann_max_id_category + 1
            
        for i in range(len(anns)):
            anns[i]['id'] += ann_max_id + 1
            new_img_id = img_id_map[anns[i]['image_id']]
            anns[i]['image_id'] = new_img_id
            anns[i]['category_id'] += ann_max_id_category + 1  # here category broken the sorted relationship
            self.ann_ids.append(anns[i]['id'])
            self.cat_ids.append(anns[i]['category_id'])
        self.merge_coco['annotations'] += anns
        self.merge_coco['categories'] += cats

    def merge(self, to_file=None):
        for dataset in self.files[1:]:
            if not isinstance(dataset, dict):
                dataset = load(dataset)
            self.update_img_ann_ids(dataset['images'], dataset['annotations'], dataset['categories'])
        if to_file:
            self.save(save=to_file)
        return self.merge_coco

    def save(self, save='merge_coco.json'):
        dump(self.merge_coco, save, indent=4)
