import cv2,os,shutil,random,glob
from xml.dom.minidom import parse
from xml.dom.minidom import Document


def save_xml(save_filename,save_width,save_height,save_depth,save_name,res,root_xml):
    """
    
    """
    doc = Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)

    folder = doc.createElement("folder")
    annotation.appendChild(folder)
    folder.appendChild(doc.createTextNode("images"))

    filename = doc.createElement("filename")
    annotation.appendChild(filename)
    filename.appendChild(doc.createTextNode("{}".format(save_filename)))

    path = doc.createElement("path")
    annotation.appendChild(path)
    path.appendChild(doc.createTextNode("path"))

    size = doc.createElement("size")
    annotation.appendChild(size)

    width = doc.createElement("width")
    size.appendChild(width)
    width.appendChild(doc.createTextNode("{}".format(int(save_width))))

    height = doc.createElement("height")
    size.appendChild(height)
    height.appendChild(doc.createTextNode("{}".format(int(save_height))))

    depth = doc.createElement("depth")
    size.appendChild(depth)
    depth.appendChild(doc.createTextNode("{}".format(save_depth)))
    for i in res:
        object = doc.createElement("object")
        annotation.appendChild(object)

        name = doc.createElement("name")
        object.appendChild(name)
        name.appendChild(doc.createTextNode("{}".format(save_name)))

        difficult = doc.createElement("difficult")
        object.appendChild(difficult)
        difficult.appendChild(doc.createTextNode("0"))

        bndbox = doc.createElement("bndbox")
        object.appendChild(bndbox)

        xmin = doc.createElement("xmin")
        bndbox.appendChild(xmin)
        xmin.appendChild(doc.createTextNode("{}".format(int(i[0]))))

        ymin = doc.createElement("ymin")
        bndbox.appendChild(ymin)
        ymin.appendChild(doc.createTextNode("{}".format(int(i[1]))))

        xmax = doc.createElement("xmax")
        bndbox.appendChild(xmax)
        xmax.appendChild(doc.createTextNode("{}".format(int(i[2]))))

        ymax = doc.createElement("ymax")
        bndbox.appendChild(ymax)
        ymax.appendChild(doc.createTextNode("{}".format(int(i[3]))))

    save_path = root_xml
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = save_path +'/'+ str(save_filename)+'.xml'
    print('save_path',save_path)
    print('res ',res)
    f = open(save_path, "w",encoding='utf-8')
    f.write(doc.toprettyxml(indent="  "))
    f.close()

def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    
    return iou



a_root = '/home/zhr/下载/睿沿/03警示牌——警示牌识别/前景字/*'
back_root = '/home/zhr/下载/睿沿/03警示牌——警示牌识别/背景'
save_pic_root = '/home/zhr/下载/睿沿/03警示牌——警示牌识别/pic'
save_xml_root = '/home/zhr/下载/睿沿/03警示牌——警示牌识别/xml'
width = 640
height = 640
depth = 3
count = 1300
save_name = 'signage_cricle_green' ##todo


a_path = glob.glob(a_root)
all_path = a_path
for i in os.listdir(back_root):
    back_path = os.path.join(back_root,i)
    back_pic = cv2.imread(back_path)
    back_pic = cv2.resize(back_pic, (width,height))
    # cv2.imshow('a',back_pic)
    # cv2.waitKey(0)
    random_n = random.sample(all_path,5)

    add_rec = []
    for i_path in random_n:
        i_pic = cv2.imread(i_path)
        i_h, i_w ,i_c = i_pic.shape
        h = random.randint(200,500)
        w = random.randint(200,500)
        i_pic = cv2.resize(i_pic,(w,h))

        r_topx = random.randint(1,width-w-10)
        r_topy = random.randint(1, height - h - 10)

        if add_rec == []:
            back_pic[r_topy:r_topy+h, r_topx:r_topx+w]=i_pic
            add_rec.append([r_topx,r_topy,r_topx+w,r_topy+h])
            continue

        all_iou = []
        for s_add in add_rec:
            iou = cal_iou(s_add, [r_topx,r_topy,r_topx+w,r_topy+h])
            all_iou.append(iou)

        if max(all_iou) >0:
            continue
        else:
            back_pic[r_topy:r_topy + h, r_topx:r_topx + w] = i_pic
            add_rec.append([r_topx, r_topy, r_topx + w, r_topy  + h])

    count+=1
    save_path = os.path.join(save_pic_root,str(count)+'.jpg')
    cv2.imwrite(save_path,back_pic)


    save_xml(count, width, height, depth, save_name, add_rec, save_xml_root)
    # cv2.imshow('a',back_pic)
    # cv2.waitKey(0)






