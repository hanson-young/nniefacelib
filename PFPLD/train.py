#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
# import logging
import os
import cv2
import numpy as np
import torch
from pfld.utils import plot_pose_cube
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
        print(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Save checkpoint to {0:}'.format(filename))

def train(train_loader, plfd_backbone, criterion, optimizer,
          epoch):
    losses = AverageMeter()
    plfd_backbone.train(True)
    for img, landmark_gt, attribute_gt, euler_angle_gt,type_flag in train_loader:
        img.requires_grad = False
        img = img.cuda()
        optimizer.zero_grad()
        attribute_gt.requires_grad = False
        attribute_gt = attribute_gt.cuda()

        landmark_gt.requires_grad = False
        landmark_gt = landmark_gt.cuda()

        euler_angle_gt.requires_grad = False
        euler_angle_gt = euler_angle_gt.cuda()

        type_flag.requires_grad = False
        type_flag = type_flag.cuda()

        plfd_backbone = plfd_backbone.cuda()

        pose, landmarks = plfd_backbone(img)
        # attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize)
        lds_loss, pose_loss = criterion(attribute_gt, landmark_gt, euler_angle_gt, type_flag,
                                                       pose, landmarks, args.train_batchsize)
        # pose_loss = pose_criterion(pose, euler_angle_gt)
        # lds_loss = pose_criterion(landmarks, landmark_gt)
        total_loss = pose_loss + lds_loss

        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item())
    return pose_loss, lds_loss


def validate(wlfw_val_dataloader, plfd_backbone, args):
    plfd_backbone.eval()

    with torch.no_grad():
        losses_NME = []
        losses_ION = []
        pose_losses_MAE = []
        for img, landmark_gt, attribute_gt, euler_angle_gt, type_flag  in wlfw_val_dataloader:
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            attribute_gt.requires_grad = False
            attribute_gt = attribute_gt.cuda(non_blocking=True)

            landmark_gt.requires_grad = False
            landmark_gt = landmark_gt.cuda(non_blocking=True)

            euler_angle_gt.requires_grad = False
            euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)

            type_flag.requires_grad = False
            type_flag = type_flag.cuda()

            plfd_backbone = plfd_backbone.cuda()

            pose, landmarks = plfd_backbone(img)

            if args.check:
                landms_tmp = landmarks.cpu().numpy()
                landms_tmp = landms_tmp.reshape(landms_tmp.shape[0], -1, 2)
                for i in range(landms_tmp.shape[0]):
                    # show result
                    show_img = np.array(np.transpose(img[i].cpu().numpy(), (1, 2, 0)))
                    show_img = (show_img * 255).astype(np.uint8)
                    np.clip(show_img, 0, 255)
                    draw = show_img.copy()
                    yaw = pose[i][0] * 180 / np.pi
                    pitch = pose[i][1] * 180 / np.pi
                    roll = pose[i][2] * 180 / np.pi
                    pre_landmark = landms_tmp[i] * [112, 112]
                    for (x, y) in pre_landmark.astype(np.int8):
                        cv2.circle(draw, (int(x), int(y)), 1, (0, 255, 0), 1)
                    draw = plot_pose_cube(draw, yaw, pitch, roll, size=draw.shape[0] // 2)

                    cv2.imshow("check", draw)
                    cv2.waitKey(0)

            landms_const = torch.tensor(-2).cuda()
            pose_const = torch.tensor(-1).cuda()

            pos1 = type_flag == landms_const

            landmarks = landmarks[pos1]
            landmark_gt = landmark_gt[pos1]
            if landmarks.shape[0] > 0:
                loss = torch.mean(
                    torch.sqrt(torch.sum((landmark_gt * 112 - landmarks  * 112)**2, dim=1))
                    )

                landmarks = landmarks.cpu().numpy()
                landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
                landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()


                error_diff = np.sum(np.sqrt(np.sum((landmark_gt - landmarks) ** 2, axis=2)), axis=1)
                interocular_distance = np.sqrt(np.sum((landmarks[:, 60, :] - landmarks[:,72, :]) ** 2, axis=1))
                # interpupil_distance = np.sqrt(np.sum((landmarks[:, 60, :] - landmarks[:, 72, :]) ** 2, axis=1))
                error_norm = np.mean(error_diff / interocular_distance)

                losses_NME.append(loss.cpu().numpy())
                losses_ION.append(error_norm)

            pos2 = type_flag == pose_const
            pose_p = pose[pos2]
            pose_t = euler_angle_gt[pos2]
            if pose_p.shape[0] > 0:
                pose_loss = torch.mean(abs(pose_t - pose_p) * 180 / np.pi)
                pose_losses_MAE.append(pose_loss.cpu().numpy())

        return np.mean(pose_losses_MAE), np.mean(losses_NME), np.mean(losses_ION)

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
            print("load %s successfully ! "%args.resume)
        except KeyError:
            plfd_backbone = torch.nn.DataParallel(plfd_backbone)
            plfd_backbone.load_state_dict(torch.load(args.resume))

    print(plfd_backbone)
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
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(0.5, 0.5, 0.5)),
    ])
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
    val_pose_loss, val_lds_nme_loss, val_lds_ion_loss = validate(wlfw_val_dataloader, plfd_backbone, args)
    print('Epoch: %d, val pose MAE:%6.4f, val lds NME:%6.4f, val lds ION:%6.4f, lr:%8.6f' % (step_index, val_pose_loss, val_lds_nme_loss, val_lds_ion_loss, cur_lr))

    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_pose_loss, train_lds_loss = train(dataloader, plfd_backbone,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth')
        save_checkpoint(plfd_backbone.state_dict(),filename)
        val_pose_loss, val_lds_nme_loss, val_lds_ion_loss = validate(wlfw_val_dataloader, plfd_backbone, args)
        if epoch in step_epoch:
            step_index += 1
            cur_lr = adjust_learning_rate(optimizer, args.base_lr, step_index)

        print('Epoch: %d, train pose loss: %6.4f, train lds loss:%6.4f, val pose MAE:%6.4f, val lds NME:%6.4f, val lds ION:%6.4f, lr:%8.6f'%(epoch, train_pose_loss, train_lds_loss, val_pose_loss, val_lds_nme_loss, val_lds_ion_loss,cur_lr))
        writer.add_scalar('data/pose_loss', train_pose_loss, epoch)
        writer.add_scalars('data/loss', {'val pose loss': val_pose_loss, 'val lds nme loss': val_lds_nme_loss, 'val lds ion loss': val_lds_ion_loss, 'train loss': train_lds_loss}, epoch)

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Trainging Template')
    # general
    parser.add_argument('-j', '--workers', default=8, type=int)

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--step', default="30,60", help="lr decay", type=str)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=80, type=int)

    #loss
    parser.add_argument('--loss', default="wing",help="strongly recommend wing loss, other loss function has been abandoned", type=str)
    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',default='./models/checkpoint/snapshot/',type=str,metavar='PATH')
    parser.add_argument('--tensorboard', default="./models/checkpoint/tensorboard", type=str)
    parser.add_argument('--resume', default='./models/checkpoint/model-wing/checkpoint_epoch_5.pth', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument('--dataroot',default='/home/unaguo/hanson/data/landmark/WFLW191104/train_data/list.txt',type=str,metavar='PATH')
    parser.add_argument('--val_dataroot',default='/home/unaguo/hanson/data/landmark/WFLW191104/test_data/list.txt',type=str,metavar='PATH')
    parser.add_argument('--train_batchsize', default=128, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    parser.add_argument('--check', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
