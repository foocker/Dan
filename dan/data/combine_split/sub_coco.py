import os
import shutil
from pycocotools.coco import COCO

from dan.data.fileio import dump
from dan.design.utils.path import mkdir_or_exist


class SubCOCO(object):
    def __init__(self, annfile, sub_categorier=None):
        """
        split a coco-like dataset to only contain the given categories, 
        a sub json file and corresponding imgs
        """
        self.coco = COCO(annfile)
        assert isinstance(sub_categorier, list), 'sub_categories should as{}'.format(['x1', 'x2']) 
        self.sub_categories = sub_categorier
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        print("All Categories is: {}".format(self.categories))
           
    def copyfile(self, imgs, src_img_folder='', dst_img_folder=''):
        if not os.path.exists(src_img_folder):
            print("{} folder is wrong".format(src_img_folder))
            return False
        mkdir_or_exist(dst_img_folder)
        print("sub imgs will be saved to {}.".format(dst_img_folder))
        for img in imgs:
            shutil.copy(os.path.join(src_img_folder, img), os.path.join(dst_img_folder, img))
        return True
    
    def subcoco(self, subannf='', src='', dst=''):
        """
        subannf: sub actegory annfile to save
        src: original imgs dir
        dst: sub imgs dir to save
        we remove the img'label that not in given subcategory, not rm the img that has others label
        HERE: can reference coco2yolo.py for a simple complete the second sence:
        for all img use annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        """
        save_dir = os.path.dirname(subannf)
        if not os.path.exists(save_dir):
            print("your save path will be here: {}".format(subannf))
            mkdir_or_exist(save_dir)
            
        cat_ids = self.coco.getCatIds(catNms=self.sub_categories)
        cat_ids_map = {v:i for i, v in enumerate(cat_ids)}
        cats = self.coco.loadCats(cat_ids)
        for cat in cats:
            cat['id'] = cat_ids_map[cat['id']]
        
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        img_ids_map = {v:i for i, v in enumerate(img_ids)}
        
        imgsname = []
        imgs = self.coco.loadImgs(img_ids)
        for img_info in imgs:
            img_info['id'] = img_ids_map[img_info['id']]
            imgsname.append(img_info['file_name'])
        
        ann_ids = self.coco.getAnnIds(imgIds=img_ids)  # add cat_ids can simple
        anns = self.coco.loadAnns(ann_ids)
        
        anns_filter = [ann for ann in anns if ann['category_id'] in cat_ids_map]
        for i, ann in enumerate(anns_filter):
            # img has more label category, remove label(when imgs small amout) or filter
            # the imgs that only contain the given subcategory(when imgs a lot)
            # for simple here is the second
            ann['image_id'] = img_ids_map[ann['image_id']]
            ann['category_id'] = cat_ids_map[ann['category_id']]
            ann['id'] = i
            
        subannfile = dict()
        subannfile["images"] = imgs
        subannfile["annotations"] = anns_filter
        subannfile["categories"] = cats
        
        dump(subannfile, subannf, indent=4)
        stat = self.copyfile(imgsname, src, dst)
        if not stat:
            print("check the input path, save is not success")
            
        
        