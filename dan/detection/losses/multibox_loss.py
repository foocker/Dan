import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.ssd_box_utils import match, log_sum_exp
GPU = True

from dan.design.builder import LOSSES


@LOSSES.register_module()
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        loc_data, conf_data, landm_data = predictions
        priors = priors
        # print('xxx', priors.shape, loc_data.shape, conf_data.shape)
        num = loc_data.size(0)
        num_priors = priors.size(0)

        # match priors (default boxes) and ground truth boxes
        # print(priors.shape, loc_data.shape)
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        if landm_data is not None:
            landm_t = torch.Tensor(num, num_priors, 10)
        else:
            landm_t = None
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            if landm_t is not None:
                landms = targets[idx][:, 4:14].data
            else:
                landms = None
            defaults = priors.data
            # print(labels.shape, truths.shape, 'content:', labels, truths, defaults.shape, defaults[100:110, :])
            match(self.threshold, truths, defaults, self.variance, labels,
                  landms, loc_t, conf_t, landm_t, idx)

            # print('hhh')
        if GPU:
            loc_t = loc_t.to('cuda')
            conf_t = conf_t.to('cuda')
            if landm_t is not None:
                landm_t = landm_t.to('cuda')

        zeros = torch.tensor(0).cuda()
        if landm_t is not None:
            # landm Loss (Smooth L1)
            # Shape: [batch, num_priors, 10]
            pos_landm = conf_t > zeros
            num_pos_landm = pos_landm.long().sum(1, keepdim=True)
            N1 = max(num_pos_landm.data.sum().float(), 1)
            # torch.Size([32, 16800, 1]) torch.Size([32, 16800, 10])  right
            # torch.Size([32, 43008, 1]) torch.Size([32, 21824, 10])  wrong
            pos_landm_idx = pos_landm.unsqueeze(pos_landm.dim()).expand_as(
                landm_data)  # [batch, pos_landm, 10]
            landm_p = landm_data[pos_landm_idx].view(-1, 10)
            landm_t = landm_t[pos_landm_idx].view(-1, 10)
            loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        else:
            loss_landm = None  # 0

        # localization Loss (Smooth L1)
        # Shape: [batch, num_priors, 4]
        pos_loc = conf_t != zeros  # same as pos_landm
        conf_t[pos_loc] = 1  # for face
        # print(pos_loc.shape, loc_data.shape)    # torch.Size([32, 16800]) torch.Size([32, 75600, 4])
        pos_loc_idx = pos_loc.unsqueeze(pos_loc.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_loc_idx].view(-1, 4)
        loc_t = loc_t[pos_loc_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_cla = log_sum_exp(batch_conf) - batch_conf.gather(
            1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_cla[pos_loc.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_cla = loss_cla.view(num, -1)
        _, loss_cla_idx = loss_cla.sort(1, descending=True)
        _, loss_cla_idx_rank = loss_cla_idx.sort(1)
        num_pos_loc = pos_loc.long().sum(1, keepdim=True)
        num_neg_loc = torch.clamp(self.negpos_ratio * num_pos_loc,
                                  max=pos_loc.size(1) - 1)
        neg_loc = loss_cla_idx_rank < num_neg_loc.expand_as(loss_cla_idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        loc_pos_idx = pos_loc.unsqueeze(2).expand_as(conf_data)
        loc_neg_idx = neg_loc.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(loc_pos_idx + loc_neg_idx).gt(0)].view(
            -1, self.num_classes)
        targets_weighted = conf_t[(pos_loc + neg_loc).gt(0)]
        # print(conf_p, targets_weighted, '\n', pos_loc.shape, torch.sum(pos_loc, dim=1))
        loss_cla = F.cross_entropy(conf_p, targets_weighted,
                                   reduction='sum')  # label

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos_loc.data.sum().float(), 1)
        loss_loc /= N
        loss_cla /= N
        if loss_landm is not None:
            loss_landm /= N1

        return loss_loc, loss_cla, loss_landm
