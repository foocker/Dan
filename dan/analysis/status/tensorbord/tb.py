from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .tbbase import Base
import numpy as np
import torch

from torchvision.ops import nms
from .utils import matplotlib_imshow

from dan.detection.inference_twodet import visualiz_inference


# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

        
class TensorboardX(Base):
    '''
    basic api reference from :https://pytorch.org/docs/stable/tensorboard.html
    '''
    def __init__(self, log_dir, classes_tuple):
        '''
        log_dir all log result will save here
        classes_tuple: classfication name map
        '''
        super(self, Base).__init__()
        self.Writer = SummaryWriter(log_dir)
        self.classes = classes_tuple
    
    def add_scalars(self, main_tag, tag_scalar_dict):
        '''
        example
        '''
        self.Writer.add_scalars(main_tag, tag_scalar_dict)
        
    def add_histogram(self, tag, values, **kwargs):
        self.Writer.add_histogram(tag, values, **kwargs)
        
    def add_images(self, tag, img_tensor, **kwargs):
        self.Writer.add_images(tag, img_tensor, **kwargs)
        
    def add_embedding(self, mat, **kwargs):
        self.Writer.add_embedding(mat, **kwargs)
        
    def add_pr_curve(self, tag, labels, predictions, **kwargs):
        self.Writer.add_pr_curve(tag, labels, predictions, **kwargs)
    
    def add_image_with_boxes(self, tag, img_tensor, box_tensor, **kwargs):
        self.Writer.add_image_with_boxes(tag, img_tensor, box_tensor, **kwargs)
        
    def images_to_probs(self, net, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = net(images)
        if output.is_cuda:
            output = output.cpu()
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        # F.softmax after logist
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(self, net, images, labels):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self.images_to_probs(net, images)
        if images.is_cuda:
            images = images.cpu()
            labels = labels.cpu()
        # plot the images in the batch, along with predicted and true labels
        if images.shape[1] == 1:
            one_channel=True
        else:
            one_channel = False
        fig = plt.figure(figsize=(10, 100))
        for idx in np.arange(10):
            ax = fig.add_subplot(1, 10, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(images[idx], one_channel=one_channel)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[preds[idx]],
                probs[idx] * 100.0,
                self.classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"))
        return fig
    
    def images_to_boxes(self, net, images, iou_threshold=0.5):
        '''
        images: batch
        logging trianning status, so should net, and images from dataloader.
        diff net has diff handfunc, here for simple, we not show pred and true label for diff
        '''
        boxes, labels, scores = net(images)  # inference_twodet.py
        nms(boxes, scores, iou_threshold=iou_threshold)  # [N, 4], [N], for batch=1
        # if boxes.is_cuda:
        #     boxes = boxes.cpu()
        #     scores = scores.cpu()
        #     labels = labels.cpu()
        
        return boxes, labels, scores
 

    def plot_boxes_preds(self, net, images, labels):
        boxes, labels_pred, scores = self.images_to_boxes(net, images, iou_threshold=0.4)
        if boxes.is_cuda:
            boxes = boxes.cpu()
            scores = scores.cpu()
            labels_pred = labels_pred.cpu()
            images = images.cpu()
        
        if images.shape[1] == 1:
            one_channel=True
        else:
            one_channel = False
            
        fig = plt.figure(figsize=(10, 100))
        for idx in np.arange(10):
            ax = fig.add_subplot(1, 10, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(images[idx], one_channel=one_channel)
            
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[labels_pred[idx]],
                scores[idx] * 100.0,
                self.classes[labels_pred[idx]]),
                        color=("green" if labels_pred[idx]==labels[idx].item() else "red"))
        return fig
    
    def status_plot(self, net, images, labels, mode="cla"):
        '''
        mode: cla, det, seg, for classfiaction detection, segmentation
        '''
        if mode == "cla":
            return self.plot_classes_preds(net, images, labels)
        elif mode == "det":
            return self.plot_boxes_preds(net, images, labels)
        elif mode == "seg":
            return None
        else:
            raise("{} not impleted at present!".format(mode))

        