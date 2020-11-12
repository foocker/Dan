from dan.analysis.datas.coco_statistic import COCOAnalysis


if __name__ == "__main__":
    annfile = '/aidata/dataset/HeiLJ/coco_format/annotations/heilj_coco_v2.json'
    base_save_path = "./data_statistics/coco-like"
    CA = COCOAnalysis(ann_file=annfile, save_dir=base_save_path)
    print(CA.save_dir, CA.catToAnns)
    # CA.stats_objs_per_cat()
    # CA.stats_objs_per_img()
    # CA.stats_size_per_cat()
    # CA.cluster_analysis(base_save_path)
    # CA.cluster_analysis(base_save_path, by_cat=True)
    CA.get_weights_for_balanced_classes()
    