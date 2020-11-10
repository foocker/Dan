import pandas as pd

import copy
import os.path as osp
import numpy as np
from collections import defaultdict

from pycocotools.coco import COCO

from .size_statistic import SizeAnalysis, load, dump
from .draw import draw_hist

from dan.design.utils.path import mkdir_or_exist

import numpy as np
from sklearn.cluster import KMeans, DBSCAN

import cv2


def k_means_cluster(data, n_clusters):
    if not isinstance(data, np.ndarray):
        raise RuntimeError('not supported data format!')
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(data)
    centroids = estimator.cluster_centers_
    return centroids


def DBSCAN_cluster(data, metric='euclidean'):
    """
    should make it more auto for paremeters
    """
    y_db_pre = DBSCAN(eps=25., min_samples=10, metric=metric).fit_predict(data)
    return y_db_pre


class COCOAnalysis(object):
    """coco-like datasets analysis"""
    def __init__(self, ann_file=None, save_dir="./data_statistics/coco-like"):
        assert ann_file is not None, "ann_file is None, should a right path"
        self.COCO = COCO(ann_file)
        self.save_dir = save_dir
        mkdir_or_exist(save_dir)
        self.sa = SizeAnalysis(self.COCO)
        self.catToAnns = defaultdict(list)

    def stats_size_per_cat(self, to_file='size_per_cat_data.json'):
        self.sa.stats_size_per_cat(to_file=osp.join(self.save_dir, to_file))

    def stats_objs_per_img(self, to_file='stats_num.json'):
        self.sa.stats_objs_per_img(to_file=osp.join(self.save_dir, to_file))

    def stats_objs_per_cat(self, to_file='objs_per_cat_data.json'):
        self.sa.stats_objs_per_cat(to_file=osp.join(self.save_dir, to_file))

    def get_weights_for_balanced_classes(self, to_file='weighted_samples.pkl'):
        weights = self.sa.get_weights_for_balanced_classes(
            to_file=osp.join(self.save_dir, to_file))
        return weights

    # TODO: is bad at present
    def cluster_analysis(self,
                         save_root,
                         name_clusters=('bbox', 'area', 'wh'),
                         n_clusters=(3, 3, 3),
                         by_cat=False):
        if by_cat:
            self._cluster_by_cat(save_root, name_clusters, n_clusters)
        assert len(name_clusters) == len(n_clusters)
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print('roidb: {}'.format(len(roidb)))
        cluster_dict = defaultdict(list)
        for entry in roidb:
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)

            objs = self.COCO.loadAnns(ann_ids)
            # Sanitize bboxes -- some are invalid
            for obj in objs:
                if 'ignore' in obj and obj['ignore'] == 1:
                    continue
                if 'area' in name_clusters:
                    cluster_dict['area'].append(obj['area'])
                if 'wh' in name_clusters:
                    cluster_dict['wh'].append(obj['bbox'][2] /
                                              float(obj['bbox'][3]))
        mkdir_or_exist(save_root)
        print('start cluster analysis...')
        for i, cluster_name in enumerate(cluster_dict.keys()):
            cluster_value = cluster_dict[cluster_name]
            assert len(cluster_value) >= n_clusters[i]
            value_arr = np.array(cluster_value)
            percent = np.percentile(value_arr, [1, 50, 99])
            value_arr = value_arr[percent[2] > value_arr]
            draw_hist(value_arr,
                      bins=1000,
                      x_label=cluster_name,
                      y_label="Quantity",
                      title=cluster_name,
                      show=False,
                      density=False,
                      save_name=osp.join(save_root, cluster_name + '.png'))
            cluster_value = np.array(value_arr).reshape(-1, 1)
            cluster_value_centers = DBSCAN_cluster(cluster_value,
                                                   metric='manhattan')
            np.savetxt(osp.join(save_root, cluster_name + '.txt'),
                       np.around(cluster_value_centers, decimals=0))
        print('cluster analysis finished!')

    def _cluster_by_cat(self,
                        save_root,
                        name_clusters=('bbox', 'area', 'wh'),
                        n_clusters=(3, 3, 3)):
        assert len(name_clusters) == len(n_clusters)
        cluster_dict = defaultdict(lambda: defaultdict(list))  # ...
        for key, ann in self.COCO.anns.items():
            cat_name = self.COCO.cats[ann['category_id']]['name']
            if 'area' in name_clusters:
                cluster_dict[cat_name]['area'].append(ann['area'])
            if 'wh' in name_clusters:
                cluster_dict[cat_name]['wh'].append(ann['bbox'][2] /
                                                    float(ann['bbox'][3]))
        mkdir_or_exist(save_root)
        for cat_name, cluster_value in cluster_dict.items():
            cluster_values = cluster_dict[cat_name]
            cluster_results = defaultdict(lambda: defaultdict(list))
            for cluster_name in cluster_values.keys():  # wh, erea
                i = name_clusters.index(cluster_name)
                if len(cluster_values[cluster_name]) < n_clusters[i]:
                    continue
                centers = k_means_cluster(np.array(cluster_values[cluster_name]).reshape(-1, 1),
                                          n_clusters=n_clusters[i])
                cluster_results[cluster_name][cat_name].append(list(centers.reshape(-1)))
            dump(cluster_results,osp.join(save_root, 'cluster_{}.json'.format(cat_name)))


def coco_data_statistic(json_path="",
                        num_threshold=[30, 150],
                        save_name="./udatacoco_statistic.csv"):
    base_static = dict()
    coco = COCO(json_path)
    cats = coco.loadCats(coco.getCatIds())
    cat_nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))

    data_statistic = []

    for cat_name in cat_nms:
        catId = coco.getCatIds(catNms=cat_name)
        imgId = coco.getImgIds(catIds=catId)
        annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
        data_statistic.append({
            "category": cat_name,
            "img_num": len(imgId),
            "box_num": len(annId)
        })

    df = pd.DataFrame(data_statistic,
                      columns=["category", "img_num", "box_num"])
    b = df["box_num"].value_counts()
    c = df["img_num"].value_counts()

    base_static["box_img_ratio"] = b / c  # dense detection or sparse

    df.to_csv(save_name, index=False)

    return base_static
