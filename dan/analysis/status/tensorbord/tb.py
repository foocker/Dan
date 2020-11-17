from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .tbbase import Base
import numpy as np
import torch

from torchvision.ops import nms
from .utils import plt_imshow


def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def clatest(testloader, net):
    '''
    simple classfication test, diff inference func
    '''
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)
    
    return test_probs, test_preds

        
class TensorboardX(Base):
    '''
    basic api reference from :https://pytorch.org/docs/stable/tensorboard.html
    '''
    def __init__(self, log_dir, classes_tuple):
        '''
        log_dir all log result will save here
        classes_tuple: classfication name map
        some **kwargs can read tensorboard.ipynb
        '''
        super(self, Base).__init__()
        self.Writer = SummaryWriter(log_dir)
        self.classes = classes_tuple
    
    def add_scalars(self, main_tag, tag_scalar_dict):
        '''
        train, val, test all in one image
        loss_train = None
        loss_test = None
        acc_train = None
        acc_test = None
        n_iter = None
        self.Writer.add_scalars('resnet_xx', {'Loss/train':loss_train,
                                        'Loss/test':loss_test,
                                        'Accuracy/train': acc_train,
                                'Accuracy/test': acc_test}, n_iter)
        '''
        self.Writer.add_scalars(main_tag, tag_scalar_dict)
        
    def add_histogram(self, tag, values, **kwargs):
        self.Writer.add_histogram(tag, values, **kwargs)
        
    def add_image(self, tag, img_tensor, **kwargs):
        '''
        input tensor make grid in one batch to a big tensor
        '''
        img_grid = make_grid(img_tensor)
        plt_imshow(img_grid, one_channel=True)   # **kwargs
        self.Writer.add_image(tag, img_grid, **kwargs)
        
    def add_embedding(self, data, targets, **kwargs):
        '''
        just show image pca or other cluster
        data: all trian data or subset
        '''
        # select random images and their target indices
        images, labels = select_n_random(data, targets)

        # get the class labels for each image
        class_labels = [self.classes[lab] for lab in labels]

        # log embeddings
        c, h, w = images.shape[1:]    # BCHW, may change
        features = images.view(-1, c * h * w)
        
        self.Writer.add_embedding(features,
                                   metadata=class_labels,
                                   label_img=images.unsqueeze(1))
        
    def add_figure(self, tag, figure, **kwargs):
        '''
        you can complete yourself plot figure, as image label, box, mask
        pred during training, or confusion matrix. 
        like plot_boxes_preds, plot_classes_preds...
        '''
        self.Writer.add_figure(tag, figure, **kwargs)
        
    def add_pr_curve(self, testloader, net, global_step=0,  **kwargs):
        '''
        Takes in a "class_index" from 0 to num_class and plots the corresponding
        precision-recall curve
        '''
        test_probs, test_preds = clatest(testloader, net)
        
        # plot all the pr curves
        for i in range(len(self.classes)):
            tensorboard_preds = test_preds == i
            tensorboard_probs = test_probs[:, i]
            self.Writer.add_pr_curve(self.classes[i], tensorboard_preds, tensorboard_probs, 
                                     global_step=global_step, **kwargs)
    
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
        show_imgs = images.shape[0] if images.shape[0] > 10 else 10
        fig = plt.figure(figsize=(10, 10*show_imgs))
        for idx in np.arange(show_imgs):
            ax = fig.add_subplot(1, show_imgs, idx+1, xticks=[], yticks=[])
            plt_imshow(images[idx], one_channel=one_channel)
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
            oc=True
        else:
            oc = False
        
        show_imgs = images.shape[0] if images.shape[0] > 10 else 10
        fig = plt.figure(figsize=(10, 10*show_imgs))
        
        for idx in np.arange(show_imgs):
            ax = fig.add_subplot(1, show_imgs, idx+1, xticks=[], yticks=[])
            ax = plt_imshow(images[idx], ax, one_channel=oc, bbox=boxes, 
                                    label=labels_pred, label_names=self.classes, score=scores)
            
        return fig
    
    
    def status_plot(self,tag, net, images, labels, mode="cla", **kwargs):
        '''
        mode: cla, det, seg, for classfiaction detection, segmentation
        '''
        if mode == "cla":
            fig = self.plot_classes_preds(net, images, labels)
        elif mode == "det":
            fig = self.plot_boxes_preds(net, images, labels)
        elif mode == "seg":
            fig =  None
        else:
            raise("{} not impleted at present!".format(mode))
        
        return self.add_figure(tag, fig, **kwargs)

        