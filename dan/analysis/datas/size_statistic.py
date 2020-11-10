import collections
from collections import defaultdict
import math
import numpy as np

from pycocotools.coco import COCO

from dan.data.fileio import load, dump

from dan.analysis.datas.draw import draw_hist, draw_class_distribution
from dan.design.utils.path import mkdir_or_exist

import os


def sort_dict(d, sort_key='key'):
    ordered_d = collections.OrderedDict()
    if sort_key == 'key':
        for key in sorted(d.keys()):
            ordered_d[key] = d[key]
    elif sort_key == 'value':
        items = d.items()
        items.sort()
        for key, value in items:
            ordered_d[key] = value
    else:
        raise TypeError('Not supported sort key:{}'.format(sort_key))
    return ordered_d


class SizeAnalysis(object):
    """
    for small objects: area < 32^2
    for medium objects: 32^2 < area < 96^2
    for large objects: area > 96^2
    see http://cocodataset.org/#detection-eval
        and https://arxiv.org/pdf/1405.0312.pdf
    """
    def __init__(self, coco, size_range=(32, 96)):
        if isinstance(coco, str):
            coco = COCO(coco)
        assert isinstance(coco, COCO)
        self.coco = coco
        self.size_range = size_range
        # self.low_limit = 0
        # self.up_limit = 100000000
        self.createIndex()

    def createIndex(self):
        self.catToDatasets = []
        catToImgs = sort_dict(self.coco.catToImgs)
        for cat, img_ids in catToImgs.items():
            img_ids = set(img_ids)
            categories = [
                cat_info for cat_info in self.coco.dataset['categories']
                if cat_info['id'] == cat
            ]
            images = [
                img_info for img_info in self.coco.dataset['images']
                if img_info['id'] in img_ids
            ]
            annotations = [
                ann_info for ann_info in self.coco.dataset['annotations']
                if ann_info['category_id'] == cat
            ]
            self.catToDatasets.append({
                'info': self.coco.dataset['info'] if 'info' in self.coco.dataset else None,
                'categories': categories,
                'images': images,
                'annotations': annotations
            })

    def stats_size_per_cat(self, to_file='size_per_cat_data.json'):
        self.cat_size = defaultdict(list)
        for cat_id, dataset in enumerate(self.catToDatasets):
            self.cat_size[dataset['categories'][0]['name']] = [
                ann_info['bbox'][2] * ann_info['bbox'][2]
                for ann_info in dataset['annotations']
            ]
        self.cat_size = dict(
            sorted(self.cat_size.items(), key=lambda item: len(item[1])))
        g2_data = []
        size_split1 = pow(self.size_range[0], 2)
        size_split2 = pow(self.size_range[1], 2)
        for cat_name, sizes in self.cat_size.items():
            data_dict = dict()
            data_dict['Category'] = cat_name
            data_dict['small'] = len(
                [size for size in sizes if size < size_split1])
            data_dict['medium'] = len(
                [size for size in sizes if size_split2 >= size > size_split1])
            data_dict['large'] = len(
                [size for size in sizes if size > size_split2])
            g2_data.append(data_dict)
        dump(g2_data, to_file)

    def stats_objs_per_img(self, to_file='stats_num.json'):
        total_anns = 0
        imgToNum = defaultdict()
        for cat_id, ann_ids in self.coco.catToImgs.items():
            imgs = set(ann_ids)
            total_anns += len(ann_ids)
            assert len(imgs) > 0
            cat_name = self.coco.cats[cat_id]['name']
            imgToNum[cat_name] = len(ann_ids) / float(len(imgs))
        imgToNum['total'] = total_anns / float(len(self.coco.imgs))
        print(imgToNum)
        dump(imgToNum, to_file)

    def stats_objs_per_cat(self, to_file='objs_per_cat_data.json'):
        cls_to_num = list()
        for cat_id in self.coco.catToImgs:
            item = dict()
            item['name'] = self.coco.cats[cat_id]['name']
            item['value'] = len(self.coco.catToImgs[cat_id])
            cls_to_num.append(item)
        dump(cls_to_num, file=to_file)
        draw_class_distribution(cls_to_num, save_name=os.path.dirname(to_file) + '/class_distribution.png')  # show

    # TODO: to fix
    def get_weights_for_balanced_classes(self, to_file='weighted_samples.pkl'):
        num_class = len(self.coco.cats)
        count_per_class = [0.] * num_class
        for cat_id, imgs in self.coco.catToImgs.items():
            print(cat_id)
            count_per_class[cat_id-1] = len(imgs)
        sort_ids = np.argsort(np.array(count_per_class))
        print(sort_ids, "sort_ids", count_per_class)
        # weight2_per_class = [0.] * len(self.coco.cats)
        # for cat_id, imgs in self.coco.catToImgs.items():
        #     weight2_per_class[cat_id-1] = len(set(imgs))
        sum_objs = sum(count_per_class)
        weight_per_class = [0.] * len(self.coco.cats)
        for i in range(len(count_per_class)):
            weight_per_class[i] = sum_objs / count_per_class[i]  # no matter what numerator is
        sum_w = sum(weight_per_class)
        for i in range(len(weight_per_class)):
            weight_per_class[i] = weight_per_class[i] / sum_w

        # log_count_per_class = [math.log(c) for c in count_per_class]
        # sum_log_count_per_class = sum(log_count_per_class)
        # log_w = [sum_log_count_per_class / log_c for log_c in log_count_per_class]
        log_max_w = math.log(max(weight_per_class))
        log_w = [math.log(w) / log_max_w for w in weight_per_class]
        sum_log_w = sum(log_w)
        for i in range(len(log_w)):
            log_w[i] = log_w[i] / sum_log_w
        from matplotlib import pyplot as plt
        plt.plot(np.array(count_per_class)[sort_ids],
                 np.array([1. / num_class] * num_class),
                 label='P=1/{}'.format(str(num_class)))
        plt.plot(np.array(count_per_class)[sort_ids],
                 np.array(weight_per_class)[sort_ids],
                 label='N*P=${C_1}$',
                 marker='o',
                 ms=4)
        plt.plot(np.array(count_per_class)[sort_ids],
                 np.array(log_w)[sort_ids][::-1],
                 label='N*log(P)=${C_2}$',
                 marker='+')
        plt.xlabel('Number of Category Instances')
        plt.ylabel('Sampling Probability')
        plt.legend(loc='best', shadow=False, fontsize=12)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.savefig(os.path.dirname(to_file) + '/P-N.png', dpi=100)
        # plt.show()    # serve don't show when has no graphicxxx
        plt.plot(np.array([1. / num_class] * num_class), label='P=1/{}'.format(num_class))
        plt.plot(np.array(weight_per_class)[sort_ids],
                 label='N*P=${C_1}$',
                 marker='o',
                 ms=4)
        plt.plot(
            np.array(log_w)[sort_ids][::-1],
            marker='+',
            label='N*log(P)=${C_2}$',
        )
        plt.xlabel('Class Index')
        plt.ylabel('Sampling Probability')
        plt.legend(loc='best', shadow=False, fontsize=12)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.savefig(os.path.dirname(to_file) + '/P-C.png', dpi=100)    
        # plt.show()
        weight = [0] * len(self.coco.imgs)
        for idx, anns in self.coco.imgToAnns.items():
            weight[idx - 1] = sum(
                [weight_per_class[ann['category_id'] - 1]
                 for ann in anns]) / len(anns)
        max_weight = max(weight)
        min_weight = min(weight)
        log_max_w = math.log(max_weight)
        log_weight = [math.log(w) / log_max_w for w in weight]
        linear_weight = [w / min_weight for w in weight]

        draw_hist(np.sort(np.array(log_weight)), bins=20, show=False, save_name=os.path.dirname(to_file) + 'hist_log_weight.png')
        draw_hist(np.sort(np.array(weight)), bins=20, show=False, save_name=os.path.dirname(to_file) + 'hist_weight.png')
        draw_hist(np.sort(np.array(linear_weight)), bins=20, show=False, save_name=os.path.dirname(to_file) + 'hist_linear_weight.png')
        sum_log_weight = sum(log_weight)
        log_weight_p = list(map(lambda x: x / sum_log_weight, log_weight))
        log_weight_p = np.sort(np.array(log_weight_p))

        def bin_data_for_same_sample_num(x, bins=10):
            bin_len = int(len(x) / bins)
            data_bin = []
            for bin_i in range(bins - 1):
                data_bin.append(
                    np.mean(x[bin_i * bin_len:(bin_i + 1) * bin_len]))
            data_bin.append(np.mean(x[(bins - 1) * bin_len:]))
            return data_bin

        def bin_data_for_same_value_interval(x, bins=10):
            x = np.sort(x)
            min_x = x[0]
            max_x = x[-1]
            bin_len = (max_x - min_x) / float(bins)
            counts = []
            means = []
            last_inds = 0
            for bin_value in np.arange(min_x + bin_len, max_x, bin_len):
                inds = 0
                for a in x:
                    if a > bin_value:
                        break
                    inds += 1
                means.append(np.mean(x[last_inds:inds]))
                counts.append(inds - last_inds)
                last_inds = inds
            return means, counts

        means, counts = bin_data_for_same_value_interval(log_weight_p,
                                                         bins=100)
        # draw_simple(x=means, y=counts)
        dump(weight, file=os.path.dirname(to_file) + 'weights.pkl')
        dump(log_weight, file=os.path.dirname(to_file) +'log_weights.pkl')
        dump(linear_weight, file=os.path.dirname(to_file) +'linear_weights.pkl')

        return weight

