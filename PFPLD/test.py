import argparse

import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import DataLoader

from data.datasets import WLFWDatasets
from pfld.pfld import PFLDInference
from pfld.utils import plot_pose_cube

import cv2

def validate(wlfw_val_dataloader, plfd_backbone, args):
    plfd_backbone.eval()

    with torch.no_grad():
        losses_NME = []
        losses_ION = []

        for img, landmark_gt, attribute_gt, euler_angle_gt, type_flag in wlfw_val_dataloader:
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

            if args.is_show:
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
                    torch.sqrt(torch.sum((landmark_gt - landmarks)**2, dim=1))
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
        print("NME", np.mean(losses_NME))
        print("ION", np.mean(losses_ION))


def main(args):
    checkpoint = torch.load(args.model_path)

    plfd_backbone = PFLDInference().cuda()

    plfd_backbone.load_state_dict(checkpoint)

    transform = transforms.Compose([transforms.ToTensor()])

    wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    validate(wlfw_val_dataloader, plfd_backbone, args)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./models/pretrained/checkpoint_robust.pth", type=str)
    parser.add_argument('--is_show', default=False, type=bool)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_dataset', default='/home/unaguo/hanson/data/landmark/WFLW191104/test_data/list.txt', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)