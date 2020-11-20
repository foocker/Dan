from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import patches
import matplotlib.font_manager
import numpy as np

import pandas as pd

def draw_class_distribution(cls_to_num, save_name='./class_distribution.png'):
    """
    绘制饼图,cls_to_num:["class_name":boxnum, ...]
    当类别数目太多，只显示前num类的统计值
    """
    #
    plt.rcParams['font.family'] = ['sans-seri']  # font windows, ubunt...
    plt.rcParams['axes.unicode_minus'] = False  # show negetive symbol
    labels = [cls_num["name"] for cls_num in cls_to_num]
    sizes = [cls_num["value"] for cls_num in cls_to_num]

    explode = tuple([0.1] * len(labels))
    fig, axes = plt.subplots(figsize=(10, 5),
                             ncols=2)  # when class_num is big, scale figsize
    ax1, ax2 = axes.ravel()
    patches, texts, autotexts = ax1.pie(sizes,
                                        explode=explode,
                                        labels=labels,
                                        autopct='%1.0f%%',
                                        shadow=False,
                                        startangle=170)
    ax1.axis('equal')
    # reset font size
    proptease = fm.FontProperties()
    proptease.set_size('xx-small')
    # font size include: ‘xx-small’,x-small’,'small’,'medium’,
    # ‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)
    ax1.set_title('Class Distribution ', loc='center')
    ax2.axis('off')
    labels_nums = [a+":" + str(b) for a, b in zip(labels, sizes)]
    ax2.set_title('category:boxes', loc='left')
    ax2.legend(patches, labels_nums, loc='center left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    
    
def draw_bar(obj_per_cat_json, xtitle='', ytitle='', savep=''):
    '''
    可视化返回的类:box_num， stats_objs_per_cat函数保存的json文件
    '''
    df = pd.read_json(obj_per_cat_json, encoding="utf-8", orient='records')
    # Draw plot
    h,w = df.shape
    fig, ax = plt.subplots(figsize=(int(h*2/3), 10), facecolor='white', dpi=80)
    ax.vlines(x=df.index, ymin=0, ymax=df.value,color='firebrick', alpha=0.7, linewidth=20)
    name_map = {v:k for k, v in enumerate(df.name)}
    print("name map is: ", name_map)

    # Annotate Text
    for i, v in enumerate(df.value):
        ax.text(i, v+1.5, round(v, 1), horizontalalignment='center')

    # Title, Label, Ticks and Ylim
    ax.set_title(xtitle, fontdict={'size':22})
    ax.set(ylabel=ytitle, ylim=(0, df.value.max()+20))
    plt.xticks(df.index, name_map.values(), rotation=60, horizontalalignment='right', fontsize=12)

    # Add patches to color the X axis labels
    p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
    p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
    fig.add_artist(p1)
    fig.add_artist(p2)
    plt.savefig(savep)
    # plt.show()
    # draw_bar(df, xtitle='Bar Chart for haihua2020 Garbage Detect', ytitle='Num Box Per Category',savep='')


def draw_hist(data,
              bins=10,
              x_label="区间",
              y_label="频数/频率",
              title="频数/频率分布直方图",
              show=True,
              save_name='hist.png',
              density=True):
    """
    绘制直方图
    data: 必选参数，绘图数据
    bins: 直方图的长条形数目，可选项，默认为10
    """
    plt.rcParams['font.family'] = ['sans-seri']  # 中文 SimHei
    plt.rcParams['axes.unicode_minus'] = False
    n, bins, patches = plt.hist(data,
                                bins=bins,
                                density=density,
                                facecolor="blue",
                                edgecolor='None')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if show:
        plt.show()
    plt.savefig(save_name, dpi=300)