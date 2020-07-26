#  -*- coding:utf-8 -*-
#  Editor      : Pycharm
#  File        : my_train.py
#  Created     : 2020/7/22 下午11:55
#  Description : train script for xunfei


import sys
import torch

from torch.autograd import Variable
import os

import chanllenge_loader
from metrics import runningScore
import models
import time
import config
import loss_metric
import utils
import my_eval


def train(args, train_loader, model, criterion, optimizer):
    model.train()

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)

    end = time.time()
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in \
            enumerate(train_loader):
        data_time.update(time.time() - end)

        imgs = Variable(imgs.cuda())
        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())
        training_masks = Variable(training_masks.cuda())

        outputs = model(imgs)
        texts = outputs[:, 0, :, :]
        kernels = outputs[:, 1:, :, :]

        selected_masks = loss_metric.ohem_batch(texts, gt_texts, training_masks)
        selected_masks = Variable(selected_masks.cuda())

        loss_text = criterion(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = Variable(selected_masks.cuda())
        for i in range(args.kernel_num - 1):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = sum(loss_kernels) / len(loss_kernels)

        loss = 0.7 * loss_text + 0.3 * loss_kernel
        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score_text = loss_metric.cal_text_score(
            texts, gt_texts, training_masks, running_metric_text)
        score_kernel = loss_metric.cal_kernel_score(
            kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0:
            output_log = '({batch}/{size}) Batch: {bt:.3f}s | ' \
                         'TOTAL: {total:.0f}min | ETA: {eta:.0f}min | ' \
                         'Loss: {loss:.4f} | Acc_t: {acc: .4f} | ' \
                         'IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=losses.avg,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'])
            print(output_log)
            sys.stdout.flush()

    return losses.avg


def save_checkpoint(state, checkpoint='checkpoint',filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def main(args):
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train_loader = chanllenge_loader.get_train_data_loader(
        args.train_annotation_path, args.img_size, args.batch_size,
        args.num_workers, args.kernel_num, args.min_scale
    )

    if args.arch == "resnet50":
        model = models.resnet50(pretrained=False, num_classes=args.kernel_num)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=args.kernel_num)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=args.kernel_num)

    if args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=0.99, weight_decay=args.weight_decay)
    else:
        raise Exception("Invalid optimizer input param.")

    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), \
            'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('Training from scratch.')

    # deploy scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.n_epoch * i) for i in [0.1, 0.3, 0.5]],
        gamma=0.8
    )

    # train
    for epoch in range(args.n_epoch):
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, args.n_epoch, optimizer.param_groups[0]['lr']))

        train_loss = train(
            args, train_loader, model, loss_metric.dice_loss, optimizer)
        # eval
        print("Start to eval...")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': args.lr,
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoint)

        logger.append(
            [optimizer.param_groups[0]['lr'], train_loss, train_te_acc,
             train_te_iou])
        scheduler.step()


if __name__ == '__main__':
    args = config.get_cfgs()
    main(args)


