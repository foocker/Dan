import os
import cv2
import json
import time
import copy
import random
import argparse
import numpy as np

# from CHEN CVer
# 需要调整的参数
ADD_SGL_CLASS_TO_MUL = False         # 是否考虑将多类图片中没有而单类图片中有的类别的图片加入.
OTHER_CLASS_IN_IMG_ACCESSED = True   # 这里是考虑, 在选择多类图片时, 当基于一个类别的标注选出图片, 图片中另外类别标注量也要考虑进去
RAND_SELECT_IMAGE_FROM_MUL = 0.05    # 当OTHER_CLASS_IN_IMG_ACCESSED为True时才起作用
SINGLE_CATEGORY_TOP_K = 0.4          # 单类与多类都有的类别中抽取前50%中随机类别
MULTI_CATEGORY_TOP_K = 0.4           # 从多类别中标注数较少的图片中选取
MAX_PADDING_BBOX_NUMS = 3            # 允许一张多类图片里面最多填充的单类BBOX个数
MAX_SELECT_IMGS_NUMS = 5             # 每一张多类图片(原版)不能被重复选的最大次数
MAX_SELECT_SGL_IMGS_NUMS = 2         # 插入到多类图片的单类图片, 同一张单类图片被抽中的最大次数.
# 固定参数
TIME_CONSUME_UPPER = 5.0             # 超时设置
IOU_TH_LOWER = 0.1
IOU_TH_UPPER = 0.3
BORDER_PIXEL = 200
# 生成数据设置项
GENERATE_IMG_NUMS = 2000             # 生成图片张数
CURRENT_IMAGES_ID = 1                # 新生成的图片从序号1开始计数
DEBUG = False
# 打印输出与保存选项
MUL_IMG_SELECTED_NUMS = 50           # 每隔50次打印各多类图片被选中的情况, 并存储在文件里, train.json文件也是同样操作.


class PictureMixupStrategy(object):
    def __init__(self, args):
        """
        fuction: mixup the bbox that come from the single_class pictures into
                 multi_class pitures
        """
        self.args = args
        self.single_cls_images_path_root = os.path.join(self.args.single_cls_root, "images")
        self.multi_cls_images_path_root = os.path.join(self.args.multi_cls_root, "images")
        self.single_cls_anno_path = os.path.join(self.args.single_cls_root, "annotations", "train.json")
        self.multi_cls_anno_path = os.path.join(self.args.multi_cls_root, "annotations", "train.json")

        # 获取json文件
        single_cls_file = open(self.single_cls_anno_path, encoding='utf-8')
        multi_cls_file = open(self.multi_cls_anno_path, encoding='utf-8')
        self.single_file_json = json.load(single_cls_file)
        self.multi_file_json = json.load(multi_cls_file)
        # 数据处理
        self._data_preprocess()
        # 记录循环过程中, 多类图片(底片)不能被重复选中的次数
        self._mul_img_select_nums = {}
        # 记录循环过程中, 单类图片(底片)不能被重复选中的次数
        self._sgl_img_select_nums = {}

    def _data_preprocess(self):
        # 为方便计算, 将数据整理成如下形式: {"image_id": bboxes_list,...}
        # 为方便存储, 将数据整理成如下形式: {"image_id": [bbox_ann_dict,...]}

        # 单类图片数据处理
        sgl_anno_json = self.single_file_json["annotations"]
        self._single_img_id_bbox_dict = {}          # {"image_id": bboxes_list,...}
        self._sgl_img_id_bbox_ann_infos_dict = {}   # {"image_id": [bbox_ann_dict,...]}
        self._sgl_img_id_to_category_id_dict = {}   # {"img_id": category_id_list}
        self._sgl_category_id_to_num_bboxes = {}    # {"category_id": num_bboxes}
        self._sgl_category_id_to_img_id_dicts = {}  # {"category_id": image_name_list}
        for ann in sgl_anno_json:
            if ann["image_id"] not in self._single_img_id_bbox_dict.keys():
                self._single_img_id_bbox_dict[ann["image_id"]] = []
                self._single_img_id_bbox_dict[ann["image_id"]].append(ann["bbox"])
            else:
                self._single_img_id_bbox_dict[ann["image_id"]].append(ann["bbox"])

            if ann["image_id"] not in self._sgl_img_id_bbox_ann_infos_dict.keys():
                self._sgl_img_id_bbox_ann_infos_dict[ann["image_id"]] = []
                self._sgl_img_id_bbox_ann_infos_dict[ann["image_id"]].append(ann)
            else:
                self._sgl_img_id_bbox_ann_infos_dict[ann["image_id"]].append(ann)

            # 以下内容是组件如下数据结构: {"img_id": category_id_list}, {"category_id": num_bboxes}
            # 和{"category_id": image_name_list}
            if ann["image_id"] not in self._sgl_img_id_to_category_id_dict.keys():
                self._sgl_img_id_to_category_id_dict[ann["image_id"]] = []
                self._sgl_img_id_to_category_id_dict[ann["image_id"]].append(ann["category_id"])
            else:
                self._sgl_img_id_to_category_id_dict[ann["image_id"]].append(ann["category_id"])

            if str(ann["category_id"]) not in self._sgl_category_id_to_num_bboxes.keys():
                self._sgl_category_id_to_num_bboxes[str(ann["category_id"])] = 0
                self._sgl_category_id_to_num_bboxes[str(ann["category_id"])] += 1
            else:
                self._sgl_category_id_to_num_bboxes[str(ann["category_id"])] += 1

            if str(ann["category_id"]) not in self._sgl_category_id_to_img_id_dicts.keys():
                self._sgl_category_id_to_img_id_dicts[str(ann["category_id"])] = []
                self._sgl_category_id_to_img_id_dicts[str(ann["category_id"])].append(ann["image_id"])
            else:
                self._sgl_category_id_to_img_id_dicts[str(ann["category_id"])].append(ann["image_id"])

        # 多类图片数据处理
        mul_anno_json = self.multi_file_json["annotations"]
        self._multi_img_id_bbox_dict = {}          # {"image_id": bboxes_list,...}
        self._mul_img_id_bbox_ann_infos_dict = {}  # {"image_id": [bbox_ann_dict,...]}
        for ann in mul_anno_json:
            if ann["image_id"] not in self._multi_img_id_bbox_dict.keys():
                self._multi_img_id_bbox_dict[ann["image_id"]] = []
                self._multi_img_id_bbox_dict[ann["image_id"]].append(ann["bbox"])
            else:
                self._multi_img_id_bbox_dict[ann["image_id"]].append(ann["bbox"])

            if ann["image_id"] not in self._mul_img_id_bbox_ann_infos_dict.keys():
                self._mul_img_id_bbox_ann_infos_dict[ann["image_id"]] = []
                self._mul_img_id_bbox_ann_infos_dict[ann["image_id"]].append(ann)
            else:
                self._mul_img_id_bbox_ann_infos_dict[ann["image_id"]].append(ann)

        self.mul_bbox_nums = 0  # 多类别图片bbox数量
        for key, bbox_list in self._multi_img_id_bbox_dict.items():
            self.mul_bbox_nums += len(bbox_list)

        # 统计类别个数和每个类别的bbox
        self.complex_bbox = {}
        self.category_num = {}
        # 创建一个拥有类别的字典self.category_num
        # 创建一个存储类别bbox的字典 self.complex_bbox
        for ann_dict in mul_anno_json:
            self.complex_bbox[ann_dict["category_id"]] = []
            self.category_num[ann_dict["category_id"]] = 0
        # 统计每个类别的个数并存储每个类别的bbox
        for ann_dict in mul_anno_json:
            w = ann_dict["bbox"][2]
            h = ann_dict["bbox"][3]
            self.complex_bbox[ann_dict["category_id"]].append([w, h])
            self.category_num[ann_dict["category_id"]] += 1

    def _extract_imgid_to_category_dict(self):
        # 针对 self._mul_img_id_bbox_ann_infos_dict进行提取, 因为self._mul_img_id_bbox_ann_infos_dict在程序
        # 运行过程中是始终保持更新的.
        # 本函数目的是为控制抽取单类图片和多类图片服务, 具体法则如下:
        # step1: 抽取单类图片时, 优先抽取多类图片中没有而单类图片中有的类别的图片, 其次, 抽取多类图片中有且标注很少
        # (top_l), 同时单类图片中也同时存在该类的图片.
        # step2: 抽取多类图片时, 选择多类img中不存在top_k类别的图片.(超时限制, 同时加少k值.)
        # 基于上述规则: 需要组件的数据结构有如下:
        # {"img_id": category_id_list}, {"category_id": num_bboxes}
        self._mul_img_id_to_category_id_dict = {}   # {"img_id": category_id_list}
        self._mul_category_id_to_num_bboxes = {}    # {"category_id": num_bboxes}
        self._mul_category_id_to_img_id_dicts = {}  # {"category_id": image_id_list}
        for image_id, ann_infos_list in self._mul_img_id_bbox_ann_infos_dict.items():
            for idx, ann_info_dict in enumerate(ann_infos_list):
                if image_id not in self._mul_img_id_to_category_id_dict.keys():
                    self._mul_img_id_to_category_id_dict[image_id] = []
                    self._mul_img_id_to_category_id_dict[image_id].append(ann_info_dict["category_id"])
                else:
                    self._mul_img_id_to_category_id_dict[image_id].append(ann_info_dict["category_id"])

                if str(ann_info_dict["category_id"]) not in self._mul_category_id_to_num_bboxes.keys():
                    self._mul_category_id_to_num_bboxes[str(ann_info_dict["category_id"])] = 0
                    self._mul_category_id_to_num_bboxes[str(ann_info_dict["category_id"])] += 1
                else:
                    self._mul_category_id_to_num_bboxes[str(ann_info_dict["category_id"])] += 1

                if str(ann_info_dict["category_id"]) not in self._mul_category_id_to_img_id_dicts.keys():
                    self._mul_category_id_to_img_id_dicts[str(ann_info_dict["category_id"])] = []
                    self._mul_category_id_to_img_id_dicts[str(ann_info_dict["category_id"])].append(ann_info_dict["image_id"])
                else:
                    self._mul_category_id_to_img_id_dicts[str(ann_info_dict["category_id"])].append(ann_info_dict["image_id"])

    def _save_json_file(self, new_ann_dict, new_image_id):
        """
        :param new_ann_dict: {"image_id": [bbox_ann_dict,...]}
        :return:
        """
        # 更新self.multi_file_json文件, 主要是更新images和annotations.
        # images下包含每一张图片的字典:{"file_name", XXX.png, "height": 1080, "weight": 1920, "image_id": XXX, "id": XXX}
        # 更新images信息
        new_image_infos_dict = {"file_name": new_image_id + ".png",
                                "height": 1080,
                                "weight": 1920,
                                "image_id": new_image_id,
                                "id": new_image_id}
        self.multi_file_json["images"].append(new_image_infos_dict)

        # 更新annotations信息, 根据new_ann_dict[new_image_id]来进行更新, 将新增bbox信息添进来
        # self.multi_file_json["annotations] -> 对应每一个bbox的详细信息:{"area": 10, "iscrowd": 0, "image_id": "xxx",
        # "bbox": [], "categord_id": 101, "id": 1, "segmentation": []}
        # new_ann_dict[new_image_id] -> 对应每一张图片所有bbox信息:[bbox_ann_dict,...], 其中每个bbox_ann_dict包含信息:
        # {"area": 10, "iscrowd": 0, "image_id": xxx, "bbox": [], "categord_id": 101, "id": 1, "segmentation": []}
        # 由于其他函数中已经完成了针对图片new_image_id的所有bbox的信息更新, 这里只需要单独再拿出来进行操作即可.
        for idx, bbox_ann_dict in enumerate(new_ann_dict[new_image_id]):
            self.multi_file_json["annotations"].append(bbox_ann_dict)

        # 将json文件存储(实时保存)
        with open(self.multi_cls_anno_path, 'w') as f:
            json.dump(self.multi_file_json, f)

        # 保存json副本文件(每隔一段时间保存副本)
        if CURRENT_IMAGES_ID % MUL_IMG_SELECTED_NUMS == 0:
            new_json_file_path = self.multi_cls_anno_path[:-5] + "_" + str(CURRENT_IMAGES_ID) + "_.json"
            with open(new_json_file_path, 'w') as f:
                json.dump(self.multi_file_json, f)

    def _compute_area(self, sgl_bbox, mul_bbox):
        """
        :function: compute the area of (bbox_a ∩ bbox_b)
        :param sgl_bbox: 单类的bbox -> (x,y,w,h)
        :param mul_bbox: 多类的bbox -> (x,y,w,h)
        :return: area between sgl_bbox and mul_bbox
        """
        sgl_bbox = [int(x) for x in sgl_bbox]
        mul_bbox = [int(x) for x in mul_bbox]
        width0, height0 = sgl_bbox[2], sgl_bbox[3]
        width1, height1 = mul_bbox[2], mul_bbox[3]

        min_x = min(sgl_bbox[0], mul_bbox[0])
        min_y = min(sgl_bbox[1], mul_bbox[1])
        max_x = max(sgl_bbox[0] + sgl_bbox[2], mul_bbox[0] + mul_bbox[2])
        max_y = max(sgl_bbox[1] + sgl_bbox[3], mul_bbox[1] + mul_bbox[3])

        width = width0 + width1 - (max_x - min_x)
        height = height0 + height1 - (max_y - min_y)

        if width <= 0 or height <= 0:
            return 0
        else:
            return width * height

    def _compute_iou(self, sgl_bbox, mul_bbox):
        """
        :function: compute the iou between bbox_a and bbox_b
        :param sgl_bbox: sgl_bbox -> (x, y, w, h)
        :param mul_bbox: mul_bbox -> (x, y, w, h)
        :return: iou between bbox_x and bbox_y
        """
        width0, height0 = sgl_bbox[2], sgl_bbox[3]
        width1, height1 = mul_bbox[2], mul_bbox[3]

        min_x = min(sgl_bbox[0], mul_bbox[0])
        min_y = min(sgl_bbox[1], mul_bbox[1])
        max_x = max(sgl_bbox[0] + sgl_bbox[2], mul_bbox[0] + mul_bbox[2])
        max_y = max(sgl_bbox[1] + sgl_bbox[3], mul_bbox[1] + mul_bbox[3])

        width = width0 + width1 - (max_x - min_x)
        height = height0 + height1 - (max_y - min_y)

        if width <= 0 or height <= 0:
            return 0

        else:
            interArea = width * height
            boxAArea = width0 * height0
            boxBArea = width1 * height1
            iou = interArea / (boxAArea + boxBArea - interArea)
            return iou

    def _get_iou_points(self, iou_th_lower, iou_th_upper, bbox_a, mulcls_bbox_list, roi_list):
        """
        :param iou_th_lower: 设定的iou最低值
        :param iou_th_upper: 设定的iou最高值
        :param bbox_a: 来自于单类图片中的bbox:(x,y,w,h)
        :param mulcls_bbox_list: 所有多类图片中的bbox:(x,y,w,h)
        :param roi_list: 多类图片标注信息的区域  [left(x1), top(y1), right(x2), down(y2)]
        """
        # 该程序可能会出现陷入无线死循环的情况, 因为这里设置超时限制.
        iou_point = None
        time_start = time.time()
        flag = 0
        while True:
            if flag == 1:
                break
            x = random.randint(roi_list[0], roi_list[2])
            y = random.randint(roi_list[1], roi_list[3])
            new_bbox_a = [x, y, bbox_a[2], bbox_a[3]]

            for bbox in mulcls_bbox_list:
                # 计算new_bbox_a与bbox_b的iou
                iou = self._compute_iou(new_bbox_a, bbox)
                if iou_th_lower <= iou <= iou_th_upper:
                    iou_point = (x, y)
                    flag = 1
                    break

            time_end = time.time()
            if (time_end - time_start) > TIME_CONSUME_UPPER:
                print("time consuming: {0} > compute time out: {1}s, restarting!".format((time_end - time_start),
                                                                                        TIME_CONSUME_UPPER))
                break

        return iou_point

    def _letterbox_image(self, image, input_dim):
        """
        :param image: 待resize的图像, 也即bbox_image
        :param input_dim: 聚类中心的尺寸
        :return: 依据短边w, 保持resize_w/image_w的进行resize.
        """
        image_w, image_h = image.shape[1], image.shape[0]
        resize_w, resize_h = [int(x) for x in input_dim]

        # 保证较短的边缩放后到正好的比例
        new_w = int(image_w * min(resize_w / image_w, resize_h / image_h))
        new_h = int(image_h * min(resize_w / image_w, resize_h / image_h))

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized_image

    def _is_reasonable(self, sgl_bbox, mul_bbox_list):
        """
        :param sgl_bbox: 单类bbox
        :param mul_bbox_list: 多类的bbox_list
        :return: True or False
        """
        # 1.计算sgl_bbox和mul_bbox_list每一个bbox的交集是否等于bbox本身
        for idx, mul_bbox in enumerate(mul_bbox_list):
            # I.完全遮挡: (mul_bbox ∩ sgl_bbox) = mul_bbox
            if self._compute_area(sgl_bbox, mul_bbox) == int(mul_bbox[2]) * int(mul_bbox[3]):
                return True
            # II.对于mul_bbox: 被遮挡了的部分大于2/5.
            elif self._compute_area(sgl_bbox, mul_bbox) > 0.4 * int(mul_bbox[2]) * int(mul_bbox[3]):
                return True
            # III.对于sgl_bbox: 如果自身超过2/5的面积在别的mul_bbox内部.
            elif self._compute_area(sgl_bbox, mul_bbox) > 0.4 * int(sgl_bbox[2]) * int(sgl_bbox[3]):
                return True

        return False

    def _update_mul_bbox(self, sgl_bbox, old_mul_bbox):
        """
        该函数的功能是: 以sgl_bbox为基准(固定), 考虑old_mul_bbox与sgl_bbox呈现的多种相对关系.
        :param sgl_bbox: 单类bbox[x,y,w,h]
        :param old_mul_bbox: 多类bbox[x,y,w,h]
        :return: new_mul_bbox
        """
        # 将[x,y,w,h] -> [x1,y1,x2,y2]
        sgl_bbox[2] = sgl_bbox[0] + sgl_bbox[2]
        sgl_bbox[3] = sgl_bbox[1] + sgl_bbox[3]
        old_mul_bbox[2] = old_mul_bbox[0] + old_mul_bbox[2]
        old_mul_bbox[3] = old_mul_bbox[1] + old_mul_bbox[3]

        # 根据sgl_bbox与old_mul_bbox之间的相对关系, 更新old_mul_bbox数据
        if sgl_bbox[0] < old_mul_bbox[0] and old_mul_bbox[2] < sgl_bbox[2]:
            # 需要更新的情况:
            if old_mul_bbox[1] < sgl_bbox[1] < old_mul_bbox[3] < sgl_bbox[3]:
                old_mul_bbox[3] = sgl_bbox[1]
            elif old_mul_bbox[3] > sgl_bbox[3] > old_mul_bbox[1] > sgl_bbox[1]:
                old_mul_bbox[1] = sgl_bbox[3]
            # 需要删除的情况,整张图片都不会更新, old_mul_bbox被sgl_bbox截断
            elif old_mul_bbox[1] < sgl_bbox[1] and sgl_bbox[3] < old_mul_bbox[3]:
                return None
            # 不需要更新的情况
            else:
                pass
        elif sgl_bbox[1] < old_mul_bbox[1] and old_mul_bbox[3] < sgl_bbox[3]:
            # 需要更新的情况
            if old_mul_bbox[0] < sgl_bbox[0] < old_mul_bbox[2] < sgl_bbox[2]:
                old_mul_bbox[2] = sgl_bbox[0]
            elif old_mul_bbox[2] > sgl_bbox[2] > old_mul_bbox[0] > sgl_bbox[0]:
                old_mul_bbox[0] = sgl_bbox[2]
            # 需要删除的情况,整张图片都不会更新, old_mul_bbox被sgl_bbox截断
            elif old_mul_bbox[0] < sgl_bbox[0] and sgl_bbox[2] < old_mul_bbox[2]:
                return None
            # 不需要更新的情况
            else:
                pass
        # 不需要更新
        else:
            pass

        # 将[x1,y1,x2,y2]->[x,y,w,h]
        new_mul_bbox = old_mul_bbox
        new_mul_bbox[2] = old_mul_bbox[2] - old_mul_bbox[0]
        new_mul_bbox[3] = old_mul_bbox[3] - old_mul_bbox[1]
        sgl_bbox[2] = sgl_bbox[2] - sgl_bbox[0]
        sgl_bbox[3] = sgl_bbox[3] - sgl_bbox[1]

        return new_mul_bbox

    def update_sgl_bbox(self, sgl_bbox, old_mul_bbox):
        """
        该函数的功能是: 以old_mul_bbox为基准(固定住), 考虑old_mul_bbox与sgl_bbox呈现的多种相对关系,
        # 为什么考虑加入这个函数: 有一些sgl_bbox插入到多类图片中时, sgl_bbox的某一边完全在mul_bbox中.
        # 这样,是不是应该将sgl_bbox也像更新_update_mul_bbox那样, 将sgl_bbox在mul_bbox中裁掉呢?
        :param sgl_bbox: 单类bbox[x,y,w,h]
        :param old_mul_bbox: 多类bbox[x,y,w,h]
        :return: new_sgl_bbox
        """
        # 将[x,y,w,h] -> [x1,y1,x2,y2]
        sgl_bbox[2] = sgl_bbox[0] + sgl_bbox[2]
        sgl_bbox[3] = sgl_bbox[1] + sgl_bbox[3]
        old_mul_bbox[2] = old_mul_bbox[0] + old_mul_bbox[2]
        old_mul_bbox[3] = old_mul_bbox[1] + old_mul_bbox[3]

        # 考虑x方向, sgl_bbox在old_mul_bbox中间的情况:
        if old_mul_bbox[0] < sgl_bbox[0] < sgl_bbox[2] < old_mul_bbox[2]:
            # 需要更新的情况
            if sgl_bbox[1] < old_mul_bbox[1] < sgl_bbox[3] < old_mul_bbox[3]:
                old_mul_bbox[3] = sgl_bbox[1]

        # 将[x1,y1,x2,y2]->[x,y,w,h]
        new_mul_bbox = old_mul_bbox
        new_mul_bbox[2] = old_mul_bbox[2] - old_mul_bbox[0]
        new_mul_bbox[3] = old_mul_bbox[3] - old_mul_bbox[1]
        sgl_bbox[2] = sgl_bbox[2] - sgl_bbox[0]
        sgl_bbox[3] = sgl_bbox[3] - sgl_bbox[1]

        return new_mul_bbox

    def _update_bbox_ann_infos(self,
                               new_bbox,
                               bbox_img_data,
                               mul_cls_image_data,
                               sgl_bbox_ann_dict,
                               mul_cls_image_name):

        # step1: 由于有遮挡出现, 这里需要重新计算出多类bbox_ann_dict中"bbox"->values信息.
        # 根据mul_cls_image_name获取所有的bbox_dict信息.
        # 注意: 这里必须深拷贝, 否则经过self._update_mul_bbox()函数后, old_mul_bbox的值会变化, 但是这里是不能改变old_mul_bbox,
        # 因为old_mul_bbox存在与原图片中, 即将生成的new_mul_bbox值是插入进新图片的.
        mul_bbox_ann_list = copy.deepcopy(self._mul_img_id_bbox_ann_infos_dict[mul_cls_image_name[:-4]])
        # 循环计算每一个mul_bbox与new_bbox之间的遮挡情况, 并更新mul_bbox的大小
        new_mul_bbox_list = []
        for idx, mul_bbox_dict in enumerate(mul_bbox_ann_list):
            new_mul_bbox = self._update_mul_bbox(new_bbox, mul_bbox_dict["bbox"])
            # 如果new_mul_bbox is None, 中断循环, 停止更新
            if new_mul_bbox is None:
                return
            new_mul_bbox_list.append(new_mul_bbox)

        # step2: 更新sgl_bbox_dict中的数据, 方便后续直接将sgl_bbox_dict插入到多类别图片的ann信息中
        # 需要更新的信息: area, image_id, bbox, id, segmentation
        # 新生成的图片不能覆盖原图片, 因此图片文件名需要更换, 更换原则: mul_cls_image_name + "sgl_cls" + str(CURRENT_IMAGES_ID)
        global CURRENT_IMAGES_ID
        copy_sgl_bbox_ann_dict = copy.deepcopy(sgl_bbox_ann_dict)
        copy_sgl_bbox_ann_dict["area"] = new_bbox[2] * new_bbox[3]
        copy_sgl_bbox_ann_dict["bbox"] = new_bbox
        copy_sgl_bbox_ann_dict["segmentation"] = []
        # 计算当前self._multi_img_id_bbox_dict或self._mul_img_id_bbox_ann_infos_dict中bbox的总数量
        copy_sgl_bbox_ann_dict["id"] = self.mul_bbox_nums + 1  # 从多类别数据
        self.mul_bbox_nums += 1
        new_image_id = mul_cls_image_name[:-4] + "_s_" + str(CURRENT_IMAGES_ID)
        copy_sgl_bbox_ann_dict["image_id"] = new_image_id

        # step3: 如果step1没有被return回去, 则说明全部计算都没有问题, 此时我们需要将
        # self._mul_img_id_bbox_ann_infos_dict 和 self._multi_img_id_bbox_dict中
        # key为mul_cls_image_name[:-4]的信息深拷贝一份. 然后进行对key为new_image_id的内容进行更新
        self._multi_img_id_bbox_dict[new_image_id] = copy.deepcopy(
                                                     self._multi_img_id_bbox_dict[mul_cls_image_name[:-4]])
        self._mul_img_id_bbox_ann_infos_dict[new_image_id] = copy.deepcopy(
                                                     self._mul_img_id_bbox_ann_infos_dict[mul_cls_image_name[:-4]])

        for idx, mul_bbox in enumerate(new_mul_bbox_list):
            bbox_area = mul_bbox[2] * mul_bbox[3]
            self._multi_img_id_bbox_dict[new_image_id][idx] = mul_bbox
            self._mul_img_id_bbox_ann_infos_dict[new_image_id][idx]["image_id"] = new_image_id
            self._mul_img_id_bbox_ann_infos_dict[new_image_id][idx]["bbox"] = mul_bbox
            self._mul_img_id_bbox_ann_infos_dict[new_image_id][idx]["area"] = bbox_area
            self._mul_img_id_bbox_ann_infos_dict[new_image_id][idx]["id"] = self.mul_bbox_nums + 1
            self.mul_bbox_nums += 1

        # 将单类bbox的信息添加进self._mul_img_id_bbox_ann_infos_dict 和 self._multi_img_id_bbox_dict
        self._multi_img_id_bbox_dict[new_image_id].append(new_bbox)
        self._mul_img_id_bbox_ann_infos_dict[new_image_id].append(copy_sgl_bbox_ann_dict)

        # 不要忘记将mixup成功的图片进行存储.
        print("当前正在生成第-{0}-张图片, 总计需要生成-{1}-张图片, 完成百分比: {2:.2%}".format(CURRENT_IMAGES_ID, GENERATE_IMG_NUMS,
                                                                                CURRENT_IMAGES_ID / GENERATE_IMG_NUMS))
        CURRENT_IMAGES_ID += 1
        # 对图像数据和json数据同时进行存储
        if not DEBUG:
            multi_cls_img_path = os.path.join(self.multi_cls_images_path_root, new_image_id + ".png")
            cv2.imwrite(multi_cls_img_path, mul_cls_image_data)
            self._save_json_file(self._mul_img_id_bbox_ann_infos_dict, new_image_id)
            print("--------------------------save successfully----------------------------")
            # 对mul_cls_image_name进行分解，获取原始多类图片的图片名
            orig_mul_img_name = mul_cls_image_name[:-4].split("_")[0]
            if orig_mul_img_name not in self._mul_img_select_nums.keys():
                self._mul_img_select_nums[orig_mul_img_name] = 0
                self._mul_img_select_nums[orig_mul_img_name] += 1
            else:
                self._mul_img_select_nums[orig_mul_img_name] += 1

            # 每生成50张，打印下采样的图片和被采样的次数:
            if CURRENT_IMAGES_ID % MUL_IMG_SELECTED_NUMS == 0:
                # 打印输出报告情况并保存:
                for img_id, selected_nums in self._mul_img_select_nums.items():
                    print("img_id: {0}, selected times: {1}.".format(img_id, selected_nums))

        # 显示用, 可以关闭.
        if DEBUG:
            draw_mul_bbox_list = self._multi_img_id_bbox_dict[new_image_id]
            for idx, bbox1 in enumerate(draw_mul_bbox_list):
                cv2.rectangle(mul_cls_image_data, (int(bbox1[0]), int(bbox1[1])),
                              (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3])), (0, 255, 0), 2)
            cv2.namedWindow("mixup_img_win", 0)
            cv2.imshow("mixup_img_win", mul_cls_image_data)
            cv2.waitKey(0)

    def _extract_sgl_img_list(self):
        self._extract_imgid_to_category_dict()
        sgl_category_set = self._sgl_category_id_to_num_bboxes.keys()
        mul_category_set = self._mul_category_id_to_num_bboxes.keys()
        # 单类图片中有, 多类图片中没有的类别集合.
        yes_sgl_class_no_mul_class_set = sgl_category_set - mul_category_set
        # 第一步: 约定优先抽取单类图片中有, 而多类图片中没有的类别的图片:
        if yes_sgl_class_no_mul_class_set and ADD_SGL_CLASS_TO_MUL:
            random_sgl_category = random.choice(list(yes_sgl_class_no_mul_class_set))
            # 第二步：获取到所有单类中含有random_sgl_category的图片的集合
            sgl_img_list = self._sgl_category_id_to_img_id_dicts[random_sgl_category]
        # 其次抽取多类图片中有, 单类图片中也有的类别, 但是这些类别在多类中占比很少的情况
        else:
            yes_sgl_class_yes_mul_class_set = sgl_category_set & mul_category_set
            # 查询yes_sgl_class_yes_mul_class_set中类在多类中的各自的占比情况
            category_ratio_by_mul = []
            for category_id in yes_sgl_class_yes_mul_class_set:
                category_ratio_by_mul.append((category_id, self._mul_category_id_to_num_bboxes[category_id]))
            # 列表按照self._mul_category_id_to_num_bboxes[category_id]排序
            category_ratio_by_mul_sorted = sorted(category_ratio_by_mul, key=lambda x: x[1])
            # 获取标注top_k的类的对应的单类的图片集合
            top_k_category_list = category_ratio_by_mul_sorted[:int(len(category_ratio_by_mul_sorted) * SINGLE_CATEGORY_TOP_K)]
            random_sgl_category = random.choice(top_k_category_list)[0]  # 随机抽取一个类
            sgl_img_list = self._sgl_category_id_to_img_id_dicts[random_sgl_category]

        return sgl_img_list, random_sgl_category

    def _mixup_picture(self, bbox, bbox_img, image_name, mlclsimage,
                       iou_th_lower, iou_th_upper, border_pixel):
        """
        :param bbox: 单类图片的标注bbox ->(x,y,w,h)
        :param bbox_img: bbox中对应的image区域
        :param image_name: 多类图片的图片名
        :param mlclsimage: 多类图片数据
        :param iou_th_lower: iou最低值
        :param iou_th_upper: iou最高值
        :return: mixup_image
        """
        orig_multi_class_img = mlclsimage.copy()
        # step1: 查找出mlclsimage中bboxes的个数
        bboxes_list = self._multi_img_id_bbox_dict[image_name]
        # 插入bbox显示
        if DEBUG:
            for idx, bbox1 in enumerate(bboxes_list):
                cv2.rectangle(orig_multi_class_img, (int(bbox1[0]), int(bbox1[1])),
                              (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3])), (0, 0, 255), 2)

        # step2: 计算多类图片的bbox区域范围
        bboxes = np.array(bboxes_list)

        box_x1 = bboxes[:, 0]
        box_y1 = bboxes[:, 1]
        box_x2 = bboxes[:, 2] + bboxes[:, 0]
        box_y2 = bboxes[:, 3] + bboxes[:, 1]

        min_x1 = np.min(box_x1)
        min_y1 = np.min(box_y1)
        min_x2 = np.max(box_x2)
        min_y2 = np.max(box_y2)

        left = max(int(min_x1) - border_pixel, 0)
        top = max(int(min_y1) - border_pixel, 0)
        right = min(int(min_x2) + border_pixel, mlclsimage.shape[1])
        down = min(int(min_y2) + border_pixel, mlclsimage.shape[0])

        ROI = [left, top, right, down]  # [x1, y1, x2, y2]

        # step3: 通过单类的bbox和ROI计算可行的点
        iou_point = self._get_iou_points(iou_th_lower, iou_th_upper, bbox, bboxes_list, ROI)
        if iou_point is None:
            return None
        # 然后将单类图片的bbox的x和y坐标换成iou_point
        new_bbox = [iou_point[0], iou_point[1], bbox[2], bbox[3]]

        # step4: 判断图片的合理性
        # 1.越界检查
        if int(iou_point[1] + bbox[3]) > mlclsimage.shape[0]:
            new_bbox[3] = mlclsimage.shape[0] - iou_point[1]
        if int(iou_point[0] + bbox[2]) > mlclsimage.shape[1]:
            new_bbox[2] = mlclsimage.shape[1] - iou_point[0]

        # 填充sgl_bbox丢失一半以上(靠近边界).(融合失败)
        if (new_bbox[2] < 0.5 * bbox[2]) or (new_bbox[3] < 0.5 * bbox[3]):
            # print("The generated picture is not available...")
            return None

        # 2.遮挡检查(融合失败)
        elif self._is_reasonable(new_bbox, bboxes_list):
            # print("Image fusion fails due to mutual occlusion...")
            return None
        # 3.输出结果(正确融合)
        # step5: 在mlclsimage的new_bbox区域, 将bbox_img部分mixup进去, mlclsimage[y1:y2, x1:x2] = bbox_img
        else:
            bbox_img = bbox_img[:int(new_bbox[3]), :int(new_bbox[2])]
            mlclsimage[int(iou_point[1]): int(iou_point[1] + new_bbox[3]),
                       int(iou_point[0]): int(iou_point[0] + new_bbox[2])] = bbox_img

            if DEBUG:
                cv2.namedWindow("orig_multiclass_img_win", 0)
                cv2.imshow("orig_multiclass_img_win", orig_multi_class_img)

            return new_bbox, bbox_img, mlclsimage

    def main_process(self):
        # step1: 随机获取一张单类图片的bbox和bbox_img
        ##############################################################
        # 本函数目的是为控制抽取单类图片和多类图片服务, 具体法则如下:
        # step1: 抽取单类图片时, 优先抽取多类图片中没有而单类图片中有的类别的图片, 其次, 抽取多类图片中有且标注很少
        # (top_l), 同时单类图片中也同时存在该类的图片.
        # step2: 抽取多类图片时, 选择多类img中不存在top_k类别的图片.(超时限制, 同时加少k值.)
        # 抽取单类图片符合条件的列表
        sgl_img_list, random_sgl_category = self._extract_sgl_img_list()
        # 获取到单类图片的集合后, 从中任意抽取一张图片
        sgl_img_name = random.choice(sgl_img_list)
        # 这里需要查询sgl_img_name被抽中了几次.
        if sgl_img_name not in self._sgl_img_select_nums.keys():
            self._sgl_img_select_nums[sgl_img_name] = 0
            self._sgl_img_select_nums[sgl_img_name] += 1
        else:
            sgl_img_selected_nums = self._sgl_img_select_nums[sgl_img_name]
            if sgl_img_selected_nums >= MAX_SELECT_SGL_IMGS_NUMS:
                return None
        # 从self._single_img_id_bbox_dict拿到bbox信息
        single_cls_bbox = self._single_img_id_bbox_dict[sgl_img_name][0]
        # 从single_cls_bbox对应的完整信息, 根据sgl_img_name来获得
        sgl_bbox_ann_dict = self._sgl_img_id_bbox_ann_infos_dict[sgl_img_name][0]
        ##############################################################
        single_cls_image_path = os.path.join(self.single_cls_images_path_root, (str(sgl_img_name) + ".png"))
        single_cls_image = cv2.imread(single_cls_image_path)
        bbox_image = single_cls_image[int(single_cls_bbox[1]):int(single_cls_bbox[1] + single_cls_bbox[3]),
                                      int(single_cls_bbox[0]):int(single_cls_bbox[0] + single_cls_bbox[2])]

        # step2: 这里对bbox_image做一定程度上的处理, 处理过程如下：
        # 1.对bbox_image做随机hflip或vflip处理, bbox不需要处理
        # 2.计算bbox与聚类中心族centeriod中哪一个的iou值最大.
        # 3.随机给定一个variance. bbox变换目标为: max_iou_centeriod + variance
        # 4.依据letterbox(只保证短边aspect_ratio)的resize法则, 对bbox进行变换.
        random_index = 1 if random.random() > 0.5 else 0
        bbox_flip_image = cv2.flip(bbox_image, random_index)
        # 获取单类图片的类别
        single_cls_category = int(random_sgl_category)
        # 如果single_cls_category可能来自于以下两种情况:
        # ①单类图片类别不在多类图片中; ②单类图片类别在多类图片中
        # 如果单类图片不在多类图片中, 暂时先不考虑①情况.
        if single_cls_category not in self.complex_bbox.keys():
            return None
        new_center_bbox_wh = random.choice(self.complex_bbox[single_cls_category])
        # step3: 用letterbox进行变换
        letterbox_single_image = self._letterbox_image(bbox_flip_image, new_center_bbox_wh)
        # 同时更新single_cls_bbox
        new_single_cls_bbox = [single_cls_bbox[0], single_cls_bbox[1],
                               letterbox_single_image.shape[1], letterbox_single_image.shape[0]]
        # step4: 随机从多类中抽取一张图片
        # 获取文件夹图片列表, 从多类别中抽取图片时,要排除掉那些本身类别占比非常高的情况.
        sorted_mul_category_id_to_num_bboxes_list = list(sorted(self._mul_category_id_to_num_bboxes.items(), key=lambda x: x[1]))
        top_k_mul_category_list = sorted_mul_category_id_to_num_bboxes_list[: int(len(sorted_mul_category_id_to_num_bboxes_list) * MULTI_CATEGORY_TOP_K)]
        random_sgl_category = random.choice(top_k_mul_category_list)[0]  # 随机选择一个类别
        mul_img_list = self._mul_category_id_to_img_id_dicts[random_sgl_category]
        multi_cls_img_name = random.choice(mul_img_list)  # 随机选择一张图片.
        # 这里需要控制一下如果拿出来的这张图片已经被重复选取了很多次，那就不要这张图片了, 重新选取, 这里主要是看
        # 对multi_cls_img_name用"_"分割, 然后进行查询含有s的个数.
        multi_cls_img_name_split_list = multi_cls_img_name.split("_")
        padding_bbox_nums = multi_cls_img_name_split_list.count("s")
        if padding_bbox_nums >= MAX_PADDING_BBOX_NUMS:
            return None
        # 查询multi_cls_img_name的最初始多类图片被选中了几次
        if multi_cls_img_name_split_list[0] in self._mul_img_select_nums.keys():
            mul_img_selected_nums = self._mul_img_select_nums[multi_cls_img_name_split_list[0]]
            if mul_img_selected_nums >= MAX_SELECT_IMGS_NUMS:
                return None
        #  样本多样性控制
        # 这里还需要进行控制, 选择出来的图片中包含的其他的类别的bbox量不能太高, 但是为了增加样本多样性, 我们应该还需要从那些其他图片中
        # 随机抽取图片(这个作为选配).
        # 这里设置一个概率阈值, 也即只有当小于该阈值情况下才考虑率从图片数据集中任意拿.
        if OTHER_CLASS_IN_IMG_ACCESSED:
            random_select_p = random.random()
            if random_select_p > RAND_SELECT_IMAGE_FROM_MUL:
                # 查询图片内部其他的bbox对应的类别
                multi_cls_img_category_list = self._mul_img_id_to_category_id_dict[multi_cls_img_name]
                # 检查该图片其他的类别的标注量, 获取阈值: num_bbox_thr, 保证不取到标注量在TOP20的类别
                num_bbox_thr = \
                    sorted_mul_category_id_to_num_bboxes_list[int(0.8 * len(sorted_mul_category_id_to_num_bboxes_list))][1]
                # 对图片中其他所有标注bbox数量进行检查, 只要有一个不合适就return None回去
                for idx, img_category_id in enumerate(multi_cls_img_category_list):
                    num_bbox_counts = self._mul_category_id_to_num_bboxes[str(img_category_id)]
                    # 默认该类别的bbox数目若在大于最高值的0.8倍, 则整张图片丢弃
                    if num_bbox_counts > num_bbox_thr:
                        return None

        multi_cls_img_path = os.path.join(self.multi_cls_images_path_root, multi_cls_img_name + ".png")
        mlclsimage = cv2.imread(multi_cls_img_path)

        # step5: 图像混合: new_data = (new_bbox, bbox_img, mlclsimage)
        new_data = self._mixup_picture(new_single_cls_bbox,
                                       letterbox_single_image,
                                       multi_cls_img_name,
                                       mlclsimage, IOU_TH_LOWER,
                                       IOU_TH_UPPER, BORDER_PIXEL)

        # step6: 整理多类图片的标注, 以图片为单位, 将ann中的bbox重新整理. 这里必须注意一个问题, 更新必须要对
        # self._multi_img_id_bbox_dict = {}和self._mul_img_id_bbox_ann_infos_dict = {}进行原地更新.
        if new_data is not None:
            self._update_bbox_ann_infos(*new_data, sgl_bbox_ann_dict, multi_cls_img_name + ".png")

        return True


def main():
    parser = argparse.ArgumentParser()
    # 单类图片文件夹root
    parser.add_argument('-single_cls_root', default="/home/pi/Desktop/work/data/competition_data/single_class",
                        help='the folder root path von single cls root')
    # 多类图片文件夹root
    parser.add_argument('-multi_cls_root', default="/home/pi/Desktop/work/data/competition_data/multi_class",
                        help='the folder root path von multi cls root')

    args = parser.parse_args()
    picturemixup = PictureMixupStrategy(args)

    while True:
        picturemixup.main_process()
        if CURRENT_IMAGES_ID > GENERATE_IMG_NUMS:
            print("恭喜您！数据生成完成。")
            break


if __name__ == '__main__':
    main()