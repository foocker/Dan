import glob, os
from dan.data.transforms.format import Labelme2COCO, COCO2YOLO, VOC2COCO


def data_format_common(style=Labelme2COCO, annotations="", output=""):
    """
    style: call calss
    annotations: labelme images and *.json files, coco json, yolo txt, voc xml files etc.
    output: save file path
    """
    print("begin data format {} transform".format(style.__name__))
    style(annotations, output)
    print("done!")
    

if __name__ == "__main__":
    # style = Labelme2COCO
    # annotations = "/aidata/dataset/cigarette/detect/"
    # # /aidata/dataset/HeiLJ/heilongjiang-Y"
    # output = "/aidata/dataset/cigarette/cigarette_coco.json"
    # annotations = "/vdata/dataset/action_detection/zhengfu/filter/"
    # output = "/aidata/dataset/HeiLJ/coco_format/annotations/chaoyang_val.json"
    annotations = "/vdata/dataset/action_detection/zhengfu/heilongjiang-Y-V2-12-1/"
    output = "/aidata/dataset/HeiLJ/coco_format/annotations/heilj_12_1.json"
    data_format_common(style=Labelme2COCO, annotations=annotations, output=output)

    # annf = "/aidata/dataset/HeiLJ/coco_format/annotations/heilj_coco_v2_sub.json"
    # imgDir="/aidata/dataset/HeiLJ/coco_format/images_sub"
    # saveDir="/aidata/dataset/HeiLJ/yolo_format_sub"
    
    # use saveDir is yolo's trianning dir is different 
    # CY = COCO2YOLO(annFile=annf, imgDir=imgDir, saveDir=saveDir)
    # CY.coco2yolo()
    
    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    # annotations_xml = '/aidata/dataset/VOC/VOCdevkit/VOC2007/Annotations/'
    # coco_to_json = '/aidata/dataset/VOC/VOCdevkit/VOC2007/AnnotationsJson/voc2007_to_coco.json'
    # data_format_common(style=VOC2COCO, annotations=annotations_xml, output=coco_to_json)
    print()
    
