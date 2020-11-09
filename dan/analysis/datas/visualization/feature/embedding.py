import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.manifold import TSNE

import umap
import random

import cv2

# https://towardsdatascience.com/tsne-vs-umap-global-structure-4d8045acba17
# https://distill.pub/2016/misread-tsne/
# 使用TSNE前必读：https://bindog.github.io/blog/2018/07/31/t-sne-tips/
# 可以用t-SNE来提出假设 不要用t-SNE得出结论
# t-SNE中集群之间的距离并不表示相似度
# t-SNE不能用于寻找离群点outlier
# 别忘了scale(perplexity)的作用
# t-SNE是在优化一个non-convex目标函数，只是局部最小
# 低维度量空间不能capture非度量的相似性，有些高维结构(距离 相似性)特征在低维是无法反映出来的


def fix_random_seeds():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def label_map(label_names):
    # dict(name=index)
    return {label_name:i for i, label_name in enumerate(label_names)}

def label_map_list(labels):
    # one list label to label_name list
    return [label_map_inverse(label) for label in labels]

def label_map_inverse(label_dict):
    # dict(index=name)
    return {v:k for k, v in label_dict.items()}

def label_color_map():
    pass

def get_features(model, dataloader):
    # from dan.data import ImageFolderPath
    # from dan.classifier.apis.inference import get_model
    # TODO
    model.eval()
    labels = []
    img_paths = []
    
    features = None
    # model just some choose layers
    for imgs, labs, path in dataloader:
        labels += labs
        img_paths += path
        with torch.no_grad():
            output  = model(imgs)
        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
        #  features = torch.cat((features, features), 1)
        #  labels = torch.cat((labels, labels), 1)
        #  features = features.cpu().numpy()
        #  features = np.concatenate((features, features), 1)
    return features, labels, img_paths


def normlize_01(x):
    vr = np.max(x) - np.min(x)
    dvr = x - np.min(x)
    return dvr / vr

def scale_image(img, max_size):
    h, w, _ = img.shape
    scale = max(1, h/max_size, w/max_size)
    img_h, img_w = int(h/scale), int(w/scale)
    
    img = cv2.resize(img, (img_w, img_h))
    return img

def n_distinct_colors(num):
    import colorsys
    """
    from :https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
    first: get n distinct HLS colors, second:translate n hls colors to RGB colors.
    """

    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    
    rgb_colors = []
    if num < 1:
        return rgb_colors
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

def draw_rectangle_class(img, label, num_class):
    h, w, _ = img.shape
    colors = n_distinct_colors(num_class)
    img = cv2.rectangle(img, (0, 0), (h-1, w-1), color=colors[label], thickness=5)  # check BGR, RGB
    
    return img

def compute_plot_coordinates(img, x, y, img_centers_area_size, offset):
    h, w, _ = img.shape

    # compute the image center coordinates on the plot
    center_x = int(img_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(img_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(w / 2)
    tl_y = center_y - int(h / 2)

    br_x = tl_x + w
    br_y = tl_y + h

    return tl_x, tl_y, br_x, br_y

def  visualize_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    # https://github.com/spmallick/learnopencv/blob/master/TSNE/tsne.py
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in zip(images, labels, tx, ty):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.show()
    

def visualize_tsne_points(tx, ty, labels):
    """
    label for setting color
    tx, ty are features(np.array) embedding 2 diemensions vector, lables are correspondding
    """
    num_class = len(set(labels))   # optimize by setting
    num_colors = np.array(n_distinct_colors(num_class)) / 255
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for label in range(num_class):
        indices = np.where(labels==label)
        tx_label = np.take(tx, indices)
        ty_label = np.take(ty, indices)
        
        ax.scatter(tx, ty, c=num_colors[label], label=label_map_inverse(label))
    
    ax.legend(loc='best')
    
    plt.savefig("tsne_points.png")
    # plt.show()


def vis_tsne_points_simple(tx, ty, labels):
    fig= plt.figure()
    ax = plt.axes()
    ax.scatter(tx, ty, alpha=0.2, c=labels, label=label_map_list(labels), cmap='viridis')
    plt.legend(loc='best')  
    
    plt.savefig("tsne_points_simple.png")
    # plt.show()
    

def visualize_tsne(tsne, images, labels, img_show=False, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = normlize_01(tx)
    ty = normlize_01(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)
    # vis_tsne_points_simple(tx, ty, labels)
    if img_show:
        # visualize the plot: samples as images
        visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)


def tsne(features):
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(features)
    tx = normlize_01(projections[:, 0])
    ty = normlize_01(projections[:, 1])
    
    plt.scatter(tx, ty)
    plt.savefig("tsne.jpg")
    # plt.show()
    

def umap(features, savename):
    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(features)
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s=5, c=features, cmap='Spectral')
    plt.title('Embedding of the training set by UMAP', fontsize=24)
    plt.savefig("{}.png".format(savename))
    

def do_tsne():
    fix_random_seeds()

    features, labels, image_paths = get_features(model=None, dataloader=None)

    tsne = TSNE(n_components=2).fit_transform(features)

    visualize_tsne(tsne, image_paths, labels, img_show=False)