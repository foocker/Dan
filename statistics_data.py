import json

import matplotlib.pyplot as plt

from dan.analysis.datas.statistics import statistic_data, statistic_anchor, statistic_fp
from dan.analysis.datas.statistics.utils import AnchorGenerator, MaxIoUAssigner 
from dan.analysis.datas.statistics.utils.parser import parse_data

# reference dan.analysis.datas.statistics.utils.parser README.md 
def parse_coco(ann_path=r'instances_train2017.json'):
    anno = parse_data(
        format='coco',
        anno_path=ann_path,
        ignore=True,
        min_size=1,
    )
    
    return anno

def parse_img_folder(imgs_folder=''):
    anno = parse_data(format='image',
                      #txt_file='data/val.txt',
                      imgs_folder=imgs_folder)
    
    return anno

def show(data):
    plt.title('my function')
    plt.hist(data, bins=100)
    """other code"""


def data_statistics_defult(save_path="./data_statistics"):
    # default statistics
    danno.default_plot()  # statistics and plot
    # danno.show()  # show the plot figure
    danno.export(save_path + '/default', save_mode='folder')  # export the analysis results to a folder
    danno.clear()  # clean the plot figure in workspace


def data_statistics_change_plot_func(save_path="./data_statistics"):

    danno.image.ratio.plot(show)
    danno.export(save_path +'/image_ratio', save_mode='folder')
    danno.clear()


def data_statistics_get_processor():
    print(danno.processor)


def data_statistics_plot_on_one_image(save_path="./data_statistics"):
    danno.figure('combine')
    danno.image.ratio.plot(show)
    danno.image.ratio.plot(plt.hist, bins=10)
    danno.image.ratio.plot(plt.hist, bins=100)
    # danno.show()  # when on serve should close it.
    danno.export(save_path + '/combine', save_mode='folder')
    danno.clear()


def anchor_statistics(save_path="./data_statistics"):
    # 1. anchor generator
    anchor_generator = AnchorGenerator(strides=[8, 16, 32, 64, 128, 256],
                                       ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
                                       octave_base_scale=4,
                                       scales_per_octave=3, )
    # 2. assigner
    assigner = MaxIoUAssigner(0.5, 0.5, ignore_iof_thr=-1)
    # 3. statistics
    aanno = statistic_anchor(anno_data, (800, 1333), anchor_generator, assigner=assigner)
    aanno.sections = [[0, 32], [32, 96], [96, 640]]
    aanno.default_plot(figsize=(15, 15), dpi=300)
    aanno.export(save_path + '/anchor_default', save_mode='folder')
    print('Anchor analysis done!')


def fp_statistics(save_path="./model_statistics"):
    """Just for detection."""
    # 1. crate dummy json file
    fake_pd = [{'image_id': 1, 'bbox': [0, 0, 5, 5], 'score': 0.5, 'category_id': 1}]
    fake_gt = {'info': {}, 'licenses': [],
               'images': [{'license': 0, 'file_name': 1, 'coco_url': '',
                           'height': 500, 'width': 400, 'date_captured': '',
                           'flickr_url': '', 'id': 1}, ],
               'annotations': [{'segmentation': [], 'area': 100, 'iscrowd': 0, 'image_id': 1,
                                'bbox': [0, 0, 10, 10], 'category_id': 1, 'id': 99, 'style': 0, 'pair_id': 0, }, ],
               'categories': [{'id': 1, 'name': '1', 'supercategory': 'tt'},
                              {'id': 2, 'name': '2', 'supercategory': 'tt'},
                              {'id': 3, 'name': '3', 'supercategory': 'tt'},
                              {'id': 4, 'name': '4', 'supercategory': 'tt'}, ]
               }
    json.dump(fake_pd, open('fake_predict.json', 'a+'))
    json.dump(fake_gt, open('fake_ground_truth.json', 'a+'))
    # 2. statistics
    fa = statistic_fp('fake_predict.json', 'fake_ground_truth.json')
    fa.default_plot()
    fa.show()
    fa.export(save_path + '/fp_analysis', save_mode='folder')
    fa.clear()
    
if __name__ == "__main__":
    anno_data = parse_coco(ann_path=r'/aidata/dataset/HeiLJ/coco_format/annotations/heilj_coco_v2.json')
    danno = statistic_data(anno_data)
    base_save_path = "./data_statistics"
    data_statistics_defult()
    data_statistics_change_plot_func()
    data_statistics_plot_on_one_image()
    anchor_statistics()
    
    