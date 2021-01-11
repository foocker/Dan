import os
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
import glob

import io, base64


def img_data_to_arr(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(Image.open(f))
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


class List2COCO(object):
    def __init__(self,list_json_path='', save_json_path="./coco.json"):
        """
        :param labelme_json: annotations path that contain the list of all labelme json file
        :param save_json_path: the path to save new json
        """
        self.list_json_path = list_json_path
        self.save_json_path = save_json_path
        self.imgname_id = dict()
        self.imgnames = set()
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.list2coco()

    def data_transfer(self):
        with open(self.list_json_path, 'r+') as f:
            list_info = json.load(f)
            
        for box_info in list_info:
            self.imgnames.add(box_info['name'])
        for i, v in enumerate(sorted(self.imgnames)):
            self.imgname_id.update({v:i})
            
        for box_info in list_info:
            img = self.image(box_info)
            if img not in self.images:
                self.images.append(img)
            label = box_info['category']    # default str(for name), but some is int(as index +1)
            if label not in self.label:
                self.label.append(label)
            x1, y1, x2, y2 = box_info['bbox']
            bbox2ploygon = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            self.annotations.append(self.annotation(bbox2ploygon, label, box_info['name']))
            self.annID += 1
            # print(self.annID)
        
        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        # for annotation in self.annotations:
        #     annotation["category_id"] = self.getcatid(annotation["category_id"])
    
    def image(self, data):
        image = {}
        height, width = data["image_height"], data["image_width"]
        image["height"] = height
        image["width"] = width
        image["id"] = self.imgname_id[data['name']]
        image["file_name"] = data["name"]

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label
        category["id"] = len(self.categories)   # 排序后的Index,0开始
        category["name"] = label
        return category

    def annotation(self, points, label, name):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        # maskUtils.frPyObjects(segm, h, w) should bbox(xywh) to (xyxy), wrong
        area = round(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))), 2)   # when seg is box, this is wrong
        annotation["segmentation"] = [list(np.asarray(points, dtype=float).flatten())]   # default is segementation
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = self.imgname_id[name]

        annotation["bbox"] = self.getbbox(points)   # from seg to bbox

        annotation["category_id"] = label -1  # self.getcatid(label)  # label-1
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        x1, y1, x2, y2 = points[0][0], points[0][1], points[2][0], points[2][1]
        return [round(x1, 2), round(y1, 2), round(x2-x1, 2), round(y2-y1, 2)]

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = sorted(self.images, key=lambda item: item['id'])
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations  # sorted(self.annotations,key=lambda item: item['image_id'])
        return data_coco

    def list2coco(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()
        
        # import numpy as np
        # for x in self.data_coco['annotations']:
        #     if isinstance(x["segmentation"][0][0], np.int64):
        #         print(x)
        #     # print(type(x['segmentation'][0][0]))
        #     # print(x['segmentation'])

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)


