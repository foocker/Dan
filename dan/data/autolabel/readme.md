https://github.com/QuickLearner171998/Auto-Label  
https://github.com/QuickLearner171998/semi-auto-image-annotation-tool/blob/master/demo.gif  
https://github.com/virajmavani/semi-auto-image-annotation-tool  
https://towardsdatascience.com/self-supervised-attention-mechanism-for-dense-optical-flow-estimation-b7709af48efd   

http://docs.dubhe.ai/docs/   :自动标注平台

## Dir orgnize
model in /snapshots/keras or /sanpshots/tensorfow or your own dir.

## Instructions
1) Select the COCO object classes for which you need suggestions from the drop-down menu and add them. Or simply click on Add all classes .  

2) Select the desired model and click on Add model.

3) Click on detect button.

4) When annotating manually, select the object class from the List and while keep it selected, select the BBox.

5) The final annotations can be found in the file annotations.csv in ./annotations/ . Also a xml file will saved.

## Usage
For MSCOCO dataset  
```
python semiauto.py  
```
