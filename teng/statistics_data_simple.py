from dan.analysis.datas.coco_statistic import COCOAnalysis
from dan.analysis.datas.draw import draw_bar
from dan.data.fileio import load, dump



if __name__ == "__main__":
    # annfile = '/aidata/dataset/HeiLJ/coco_format/annotations/heilj_coco_v2.json'
    # base_save_path = "./data_statistics/coco-like"
    annfile = '/aidata/dataset/haihua_2020_detect/train/train_add_id.json'
    base_save_path = "./data_statistics/coco-like_haihua2020"
    CA = COCOAnalysis(ann_file=annfile, save_dir=base_save_path)
    print(CA.save_dir, CA.catToAnns)
    CA.stats_objs_per_cat()
    CA.stats_objs_per_img()
    CA.stats_size_per_cat()
    # CA.cluster_analysis(base_save_path)
    # CA.cluster_analysis(base_save_path, by_cat=True)
    # CA.get_weights_for_balanced_classes()
    
    # dataset = load(annfile)
    # # for i, ann in enumerate(dataset['annotations']):
    # #     ann['id'] = i
    #     # ann.update({'id':i})
    # for i, img in enumerate(dataset['images']):
    #     img['id']= i
        
    # dump(dataset, '/aidata/dataset/haihua_2020_detect/train/train_add_id.json')
    
        
