from pycocotools.coco import COCO
from shutil import copyfile
import os

from dan.design.utils.path import mkdir_or_exist


class COCO2YOLO(object):
    def __init__(self, annFile, imgDir='', saveDir='', mode='train'):
        """
        imgDir: original img path
        saveDir: data dir save for yolo, not contain images/trian ...
        mode: train, val, trianval, etc
        """
        self.coco = COCO(annFile)
        self.imgDir = imgDir
        self.saveDir = saveDir
        self.save_trainimg_dir = os.path.join(self.saveDir, "images/{}".format(mode))
        self.save_trainlabel_dir = os.path.join(self.saveDir, "labels/{}".format(mode))

        mkdir_or_exist(self.save_trainimg_dir)
        mkdir_or_exist(self.save_trainlabel_dir)
        
        # self.coco2yolo()
        
    def coco2yolo(self):
        """
        """
        catIds = self.coco.getCatIds()
        catsInfo = self.coco.loadCats(catIds)
        nms=[cat['name'] for cat in catsInfo]
        print('COCO categories: {}'.format(nms))

        # imgIds = self.coco.getImgIds(catIds=catIds) # may empty
        imgIds = self.coco.getImgIds()
        imgs = self.coco.loadImgs(imgIds)
        for img in imgs: 
            fName = img["file_name"]
            imgPath = os.path.join(self.imgDir, fName)
            newfName = fName.split("/")[-1]    # orginal labelme2coco may add a sencond dir
            newPath = os.path.join(self.save_trainimg_dir,  newfName)
            txtFile = os.path.join(self.save_trainlabel_dir, os.path.splitext(newfName)[0] + '.txt')
            copyfile(imgPath, newPath)
            dw = 1. / img['width']
            dh = 1. / img['height']
            annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            
            with open(txtFile, 'w+') as file:
                for ann in anns:
                    box = ann["bbox"]
                    x = box[0] + box[2]/2
                    y = box[1] + box[3]/2
                    w = box[2]
                    h = box[3]
                    x = round(x * dw, 3)
                    y = round(y * dh, 3)
                    w = round(w * dw, 3)
                    h = round(h * dh, 3)
                    classId = ann['category_id']
                    strs = "{} {} {} {} {}\n".format(classId, x, y, w, h)
                    file.write(strs)
        
        print("coco to yolo done!")