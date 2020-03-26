#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
import os

import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from data.datasets import WLFWDatasets
from pfld.pfld import PFLDInference
from pfld.loss import PFLDLoss, MSELoss, SmoothL1, WingLoss
from pfld.utils import AverageMeter

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Save checkpoint to {0:}'.format(filename))

def train(train_loader, plfd_backbone, criterion, optimizer,
          epoch):
    losses = AverageMeter()
    plfd_backbone.train(True)
    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        img.requires_grad = False
        img = img.cuda()
        optimizer.zero_grad()
        attribute_gt.requires_grad = False
        attribute_gt = attribute_gt.cuda()

        landmark_gt.requires_grad = False
        landmark_gt = landmark_gt.cuda()

        euler_angle_gt.requires_grad = False
        euler_angle_gt = euler_angle_gt.cuda()

        plfd_backbone = plfd_backbone.cuda()

        pose, landmarks = plfd_backbone(img)
        # attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize)
        lds_loss, pose_loss = criterion(attribute_gt, landmark_gt, euler_angle_gt,
                                                       pose, landmarks, args.train_batchsize)
        # pose_loss = pose_criterion(pose, euler_angle_gt)
        # lds_loss = pose_criterion(landmarks, landmark_gt)
        total_loss = pose_loss + lds_loss

        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item())
    return pose_loss, lds_loss


def validate(wlfw_val_dataloader, plfd_backbone, criterion,
             epoch):
    plfd_backbone.eval()

    pose_losses = []
    landmark_losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in wlfw_val_dataloader:
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            attribute_gt.requires_grad = False
            attribute_gt = attribute_gt.cuda(non_blocking=True)

            landmark_gt.requires_grad = False
            landmark_gt = landmark_gt.cuda(non_blocking=True)

            euler_angle_gt.requires_grad = False
            euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)

            plfd_backbone = plfd_backbone.cuda()

            pose, landmark = plfd_backbone(img)

            landmark_loss = torch.mean(abs(landmark_gt - landmark) * 112)
            landmark_losses.append(landmark_loss.cpu().numpy())

            pose_loss = torch.mean(abs(pose - euler_angle_gt) * 180 / np.pi)
            pose_losses.append(pose_loss.cpu().numpy())
    return np.mean(pose_losses), np.mean(landmark_losses)

def adjust_learning_rate(optimizer, initial_lr, step_index):

    lr = initial_lr * (0.1 ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main(args):
    print_args(args)

    plfd_backbone = PFLDInference().cuda()
    if args.resume:
        try:
            plfd_backbone.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
            logging.info("load %s successfully ! "%args.resume)
        except KeyError:
            plfd_backbone = torch.nn.DataParallel(plfd_backbone)
            plfd_backbone.load_state_dict(torch.load(args.resume))

    step_epoch = [int(x) for x in args.step.split(',')]
    if args.loss == 'mse':
        criterion = MSELoss()
    elif args.loss == 'sommthl1':
        criterion = SmoothL1()
    elif args.loss == 'wing':
        criterion = WingLoss()
    elif args.loss == 'pfld':
        criterion = PFLDLoss()
    cur_lr = args.base_lr
    optimizer = torch.optim.Adam(
        plfd_backbone.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay)


    # SGD optimizer is very sensitive to the magnitude of loss,
    # which is likely to be non convergent or nan, I recommend Adam.
    # optimizer = torch.optim.SGD(plfd_backbone.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)

    train_transform = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.ToTensor()])
    wlfwdataset = WLFWDatasets(args.dataroot, train_transform)
    dataloader = DataLoader(
        wlfwdataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)
    val_transform = transforms.Compose([transforms.ToTensor()])
    wlfw_val_dataset = WLFWDatasets(args.val_dataroot, val_transform)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    step_index = 0
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_pose_loss, train_lds_loss = train(dataloader, plfd_backbone,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth')
        save_checkpoint(plfd_backbone.state_dict(),filename)
        val_pose_loss, val_lds_loss = validate(wlfw_val_dataloader, plfd_backbone,
                            criterion, epoch)
        if epoch in step_epoch:
            step_index += 1
            cur_lr = adjust_learning_rate(optimizer, args.base_lr, step_index)

        print('Epoch: %d, train pose loss: %6.4f, train lds loss:%6.4f, val pose MAE:%6.4f, val lds MAE:%6.4f, lr:%8.6f'%(epoch, train_pose_loss, train_lds_loss, val_pose_loss, val_lds_loss,cur_lr))
        writer.add_scalar('data/pose_loss', train_pose_loss, epoch)
        writer.add_scalars('data/loss', {'val pose loss': val_pose_loss, 'val lds loss': val_lds_loss, 'train loss': train_lds_loss}, epoch)

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Trainging Template')
    # general
    parser.add_argument('-j', '--workers', default=8, type=int)

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--step', default="50,100,130", help="lr decay", type=str)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=150, type=int)

    #loss
    parser.add_argument('--loss', default="wing",help="mse, pfld, sommthl1 or wing, strongly recommend wing loss", type=str)
    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',default='./models/checkpoint/snapshot/',type=str,metavar='PATH')
    parser.add_argument('--tensorboard', default="./models/checkpoint/tensorboard", type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument('--dataroot',default='/home/unaguo/hanson/data/landmark/WFLW191104/train_data/list.txt',type=str,metavar='PATH')
    parser.add_argument('--val_dataroot',default='/home/unaguo/hanson/data/landmark/WFLW191104/test_data/list.txt',type=str,metavar='PATH')
    parser.add_argument('--train_batchsize', default=128, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
