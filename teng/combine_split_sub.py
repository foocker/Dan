from dan.data.combine_split import MergeCOCO, SplitDataset, SubCOCO


if __name__ == "__main__":
    ann_file1 = '/aidata/dataset/HeiLJ/coco_format/annotations/heilj_coco_v2.json'
    ann_file2 = '/aidata/dataset/cigarette/cigarette_coco.json'
    merged_file = '/aidata/dataset/cigarette/merge_coco/cigarette_heilj_merge.json'
    splited_file = '/aidata/dataset/cigarette/merge_coco/cigarette_heilj_split.json'
    
    # Merge
    # dataset_files = [ann_file1, ann_file2]
    # merge_coco = MergeCOCO(dataset_files)
    # merge_coco.merge()
    # merge_coco.save(save=merged_file)
    
    # # Split
    # split_data = SplitDataset(merged_file)
    # split_data.split_cats()
    # split_data.save_cat_datasets(to_file=splited_file)
    
    # Sub
    subcoco = SubCOCO(ann_file1, sub_categorier=['work_table', 'worker'])
    sub_file = '/aidata/dataset/HeiLJ/coco_format/annotations/heilj_coco_v2_sub.json'
    srcp = '/aidata/dataset/HeiLJ/coco_format/images'
    dstp = '/aidata/dataset/HeiLJ/coco_format/images_sub'
    subcoco.subcoco(sub_file, src=srcp, dst=dstp)
    
