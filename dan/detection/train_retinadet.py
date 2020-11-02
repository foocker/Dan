from __future__ import print_function
import os
import torch
from torch import optim
from torch.utils import data
# from dan.data.wider_face import WiderFaceDetection, detection_collate
from dan.data.transforms import preproc, get_augumentation
from dan.design import Config
from dan.detection.losses.multibox_loss import MultiBoxLoss
from dan.detection.core.anchors.prior_box import PriorBox
from dan.design.builder import build_loss, build_detector
from dan.data.coco import CocoDataset, coco_collate
from dan.detection.detectors import RetinaDet  # ? why must import at here

import time
import datetime
import math

cfg = Config.fromfile('/root/Codes/Dan/dan/detection/config/face_config.py')

# print('base cfg name is {}.'.format(cfg['name']))
rgb_mean = cfg.train_cfg.xx['rgb_means']
num_classes = cfg.train_cfg.xx['num_classes']
img_dim = cfg.train_cfg.xx['image_size']
num_gpu = cfg.train_cfg.xx['ngpu']
batch_size = cfg.train_cfg.xx['batch_size']
max_epoch = cfg.train_cfg.xx['epoch']
gpu_train = cfg.train_cfg.xx['gpu_train']

num_workers = cfg.train_cfg.xx['num_workers']
momentum = cfg.train_cfg.xx['momentum']
weight_decay = cfg.train_cfg.xx['weight_decay']
initial_lr = cfg.train_cfg.xx['lr']
gamma = cfg.train_cfg.xx['gamma']
training_label = cfg.train_cfg.xx['training_label']
training_dataset = cfg.train_cfg.xx['training_dataset']
save_weights = cfg.train_cfg.xx['save_weights']
weights_label = cfg.train_cfg.xx['weights_label']
resume_epoch = cfg.train_cfg.xx['resume_epoch']

net = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
# net = cfg['net'](cfg=cfg, phase='train', scene='general')
net = net.to('cuda')

if not os.path.exists(save_weights):
    os.mkdir(save_weights)

# resume

# distribute

optimizer = optim.SGD(net.parameters(),
                      lr=initial_lr,
                      momentum=momentum,
                      weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.3, False)

# criterion = build_loss(cfg)

priorboxs = PriorBox(cfg.train_cfg.xx, image_size=(img_dim, img_dim))
# priorboxs = cfg['pribox'](cfg, image_size=(img_dim, img_dim), box_normalize=False)    # when data loader not normalize box
with torch.no_grad():
    priors = priorboxs.forward()
    priors = priors.to('cuda')  # change when distribute completed
    # print(priors.shape)


def train():
    net.train()
    epoch = 0 + resume_epoch
    print('Loading Dataset....')

    # dataset = WiderFaceDetection(training_label, training_dataset, preproc(img_dim, rgb_mean))
    # dataset = WiderFaceDetection(training_label, training_dataset, preproc(img_dim, rgb_mean))
    # dataset = CocoDataset(training_dataset, set_name='car_coco_train', transform=get_augumentation(phase='train', width=640, height=640))
    dataset = CocoDataset(training_dataset,
                          set_name='annotations',
                          transform=get_augumentation(phase='train',
                                                      width=640,
                                                      height=640,
                                                      ft='coco'))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg.train_cfg.xx['decay1'] * epoch_size,
                  cfg.train_cfg.xx['decay2'] * epoch_size)
    step_index = 0

    if resume_epoch > 0:
        start_iter = resume_epoch * epoch_size
    else:
        start_iter = 0

    for it in range(start_iter, max_iter):
        if it % epoch_size == 0:
            epoch_iterator = iter(
                data.DataLoader(dataset,
                                batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                collate_fn=coco_collate))
            if (epoch % 80 == 0
                    and epoch > 0) or (epoch % 80 == 0
                                       and epoch > cfg.train_cfg.xx['decay1']):
                torch.save(
                    net.state_dict(),
                    os.path.join(
                        save_weights, cfg.train_cfg.xx['name'] +
                        weights_label + 'epoch_' + str(epoch) + '.pth'))
            epoch += 1

        load_start = time.time()
        if it in stepvalues:
            step_index += 1
        lr = adujst_learning_rate(optimizer, gamma, epoch, step_index, it,
                                  epoch_size)

        # load train data
        imgs, targets = next(epoch_iterator)  # batch or epoch_iterator?
        # print(targets)
        imgs = imgs.to('cuda')
        targets = [anno.to('cuda') for anno in targets]

        out = net(imgs)
        # print(out[0].shape)

        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        if loss_landm is not None:
            loss = cfg.train_cfg.xx['loc_weight'] * loss_l + loss_c + loss_landm
        else:
            loss = cfg.train_cfg.xx['loc_weight'] * loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_end = time.time()
        batch_time = load_end - load_start
        eta = int(batch_time * (max_iter - it))
        # print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || '
        #       'LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(
        #     epoch, max_epoch, (it % epoch_size) + 1, epoch_size, it + 1, max_iter,
        #     loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        print(
            'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f}  || '
            'LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(
                epoch, max_epoch, (it % epoch_size) + 1, epoch_size, it + 1,
                max_iter, loss_l.item(), loss_c.item(), lr, batch_time,
                str(datetime.timedelta(seconds=eta))))

    torch.save(
        net.state_dict(),
        os.path.join(save_weights, cfg.train_cfg.xx['name'] + '_Final.pth'))


def adujst_learning_rate(optimizer, gamma, epoch, step_index, iteration,
                         epoch_size):
    """Sets the learning rate
     Adapted from PyTorch Imagenet example:
     https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size *
                                                       warmup_epoch)
    else:
        lr = initial_lr * (gamma**step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == "__main__":
    train()
