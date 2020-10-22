from pycocotools.coco import COCO
import pandas as pd

# dataDir='/aidata/dataset/cigarette/test/cig_multi_3/'
# dataType='val2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

annFile = '/aidata/dataset/cigarette/instances_default.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cat_nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))

# with open("cigrette_label_statistic.txt") as f:
#     pass

data_statistic = []

# 统计各类的图片数量和标注框数量
for cat_name in cat_nms:
    catId = coco.getCatIds(catNms=cat_name)
    imgId = coco.getImgIds(catIds=catId)
    annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
    data_statistic.append({"category":cat_name, "img_num":len(imgId), "box_num":len(annId)})

    # print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))

# print(data_statistic)
df = pd.DataFrame(data_statistic, columns=["category", "img_num", "box_num"])
# print(df)
c = df["box_num"].value_counts()
print(sum(df["box_num"]))
print(sum(df["img_num"]))
# print(df["category"])
print(df[df["box_num"] ==0].shape)
box_geq10 = df[df["box_num"] > 10]
box_geq20 = df[df["box_num"] > 20]
box_geq30 = df[df["box_num"] > 30]
box_geq150 = df[df["box_num"] > 150]

box_geq10_category = set(box_geq10["category"])
box_geq20_category = set(box_geq20["category"])
box_geq30_category = set(box_geq30["category"])

box_geq_12_category = box_geq10_category.difference(box_geq20_category)
box_geq23_category = box_geq20_category.difference(box_geq30_category)

print(box_geq150.shape, len(box_geq10_category), len(box_geq20_category), len(box_geq30_category))

print(box_geq_12_category, len(box_geq_12_category), "\n", box_geq23_category, len(box_geq23_category))

# write and save
# df.to_csv("./cigrette_multi3_statisti.csv", index=False)
# pd.read_csv('data.csv', header=None)