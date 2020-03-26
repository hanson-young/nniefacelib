import argparse

import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import DataLoader

from data.datasets import WLFWDatasets
from pfld.pfld import PFLDInference
from pfld.utils import plot_pose_cube

import cv2

def validate(wlfw_val_dataloader, plfd_backbone):
    plfd_backbone.eval()

    with torch.no_grad():
        losses = []
        losses_ION = []

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

            pose, landmarks = plfd_backbone(img)

            loss = torch.mean(
                torch.sqrt(torch.sum((landmark_gt - landmarks)**2, dim=1))
                )

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()
            error_diff = np.sum(np.sqrt(np.sum((landmark_gt - landmarks) ** 2, axis=2)), axis=1)
            interocular_distance = np.sqrt(np.sum((landmarks[:, 60, :] - landmarks[:,72, :]) ** 2, axis=1))
            # interpupil_distance = np.sqrt(np.sum((landmarks[:, 60, :] - landmarks[:, 72, :]) ** 2, axis=1))
            error_norm = np.mean(error_diff / interocular_distance)

            # show result 
            show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
            show_img = (show_img * 255).astype(np.uint8)
            np.clip(show_img, 0, 255)

            pre_landmark = landmarks[0] * [112, 112]

            # cv2.imwrite("xxx.jpg", show_img)
            # img_clone = cv2.imread("xxx.jpg")
            draw  = show_img.copy()
            yaw = pose[0][0] * 180 / np.pi
            pitch = pose[0][1] * 180 / np.pi
            roll = pose[0][2] * 180 / np.pi
            for (x, y) in pre_landmark.astype(np.int8):
                # print("x:{0:}, y:{1:}".format(x, y))
                cv2.circle(draw, (int(x), int(y)), 1, (0,255,0), 1)
            draw = plot_pose_cube(draw, yaw, pitch, roll, size=draw.shape[0] // 2)

            # cv2.imshow("xx.jpg", draw)
            # cv2.waitKey(0)

        losses.append(loss.cpu().numpy())
        losses_ION.append(error_norm)

        print("NME", np.mean(losses))
        print("ION", np.mean(losses_ION))


def main(args):
    checkpoint = torch.load(args.model_path)

    plfd_backbone = PFLDInference().cuda()

    plfd_backbone.load_state_dict(checkpoint)

    transform = transforms.Compose([transforms.ToTensor()])

    wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset, batch_size=8, shuffle=False, num_workers=0)

    validate(wlfw_val_dataloader, plfd_backbone)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./models/pretrained/checkpoint_epoch_final.pth", type=str)
    parser.add_argument('--test_dataset', default='/home/unaguo/hanson/data/landmark/WFLW191104/test_data/list.txt', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)