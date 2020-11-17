import matplotlib.pyplot as plt
import numpy as np


# helper function to show an image
        
def plt_imshow(img, ax, one_channel=False, bbox=None, label=None, label_names=None, score=None, mask=None):
    '''
    show in trainning for classfication ,detection or instance segementation
    '''
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        ax.imshow(npimg, cmap="Greys")
    else:
        # CHW -> HWC
        # ax.imshow(img.astype(np.uint8))
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        
    if bbox is not None:
        if len(bbox) == 0:
            return ax

        for i, bb in enumerate(bbox):
            xy = (bb[1], bb[0])
            height = bb[2] - bb[0]
            width = bb[3] - bb[1]
            ax.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor='red', linewidth=2))

            caption = list()

            if label is not None and label_names is not None:
                lb = label[i]
                if not (0 <= lb < len(label_names)):  # modfy here to not add backgroud
                    raise ValueError('No corresponding name is given')
                caption.append(label_names[lb])
            if score is not None:
                sc = score[i]
                caption.append('{:.2f}'.format(sc))

            if len(caption) > 0:
                ax.text(bb[1], bb[0],
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    
    return ax