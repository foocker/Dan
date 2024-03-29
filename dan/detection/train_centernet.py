import os

import torch
from  torch.utils.data import DataLoader
from dan.detection.center_opts import opts
from dan.detection.apis.centernet.ctdet import CtdetTrainer
from dan.data import DataParallel, CigBox
from dan.detection.utils import load_model, save_model
from dan.detection.detectors.center import CtdetDetector   # for registry

from dan.design import Config
from dan.design.builder import build_heatmap


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = CigBox
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    #   logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    # model = create_model(opt.arch, opt.heads, opt.head_conv)
    # cfg = Config('./centernet/configs/res18_centernet.py')
    cfg = Config.fromfile(opt.config)
    # print(opt.config)
    # print(cfg.model)
    # here we only use detector registry, not contain data, pipleline, loss, path, etc...
    model = build_heatmap(cfg.model)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model,
                                                   optimizer, opt.resume,
                                                   opt.lr, opt.lr_step)

    Trainer = CtdetTrainer
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = DataLoader(Dataset(opt, 'val'),
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True)

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_loader = DataLoader(Dataset(opt, 'train'),
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers,
                                               pin_memory=True,
                                               drop_last=True)

    print('Starting training...', 'epoch is: ', opt.num_epochs, opt.dataset.default_resolution)
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        # logger.write('epoch: {} |'.format(epoch))
        # for k, v in log_dict_train.items():
        #   logger.scalar_summary('train_{}'.format(k), v, epoch)
        #   logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
        #   for k, v in log_dict_val.items():
        # logger.scalar_summary('val_{}'.format(k), v, epoch)
        # logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch,
                           model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch,
                       model, optimizer)
        # logger.write('\n')
        if epoch in opt.lr_step:
            save_model(
                os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                epoch, model, optimizer)
            lr = opt.lr * (0.1**(opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


#   logger.close()

def train_centernet():
    opt = opts().parse()
    main(opt)
    

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)

