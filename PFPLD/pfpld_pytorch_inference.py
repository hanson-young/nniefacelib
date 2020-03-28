import argparse
import numpy as np
import torch
from torchvision import transforms
import cv2
import os
import glob
from pfld.pfld import PFLDInference
from hdface.hdface import hdface_detector
from pfld.utils import plot_pose_cube
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main(args):
    det = hdface_detector(use_cuda=False)
    checkpoint = torch.load(args.model_path)
    plfd_backbone = PFLDInference().cuda()
    plfd_backbone.load_state_dict(checkpoint)
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.cuda()
    transform = transforms.Compose([transforms.ToTensor()])
    root = args.images_path


    path_list = glob.glob(os.path.join(root, "*.jpg"))
    # cap = cv2.VideoCapture("")
    for img_path in path_list:
        img = cv2.imread(img_path)

        height, width = img.shape[:2]
        img_det = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = det.detect_face(img_det)
        for i in range(len(result)):
            box = result[i]['box']
            cls = result[i]['cls']
            pts = result[i]['pts']
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 25))
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size_w = int(max([w, h])*0.9)
            size_h = int(max([w, h]) * 0.9)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size_w//2
            x2 = x1 + size_w
            y1 = cy - int(size_h * 0.4)
            y2 = y1 + size_h

            left = 0
            top = 0
            bottom = 0
            right = 0
            if x1 < 0:
                left = -x1
            if y1 < 0:
                top = -y1
            if x1 >= width:
                right = x2 - width
            if y1 >= height:
                bottom = y2 - height

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            cropped = img[y1:y2, x1:x2]
            print(top, bottom, left, right)
            cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
            
            cropped = cv2.resize(cropped, (112, 112))

            input = cv2.resize(cropped, (112, 112))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = transform(input).unsqueeze(0).cuda()
            pose, landmarks = plfd_backbone(input)
            poses = pose.cpu().detach().numpy()[0] * 180 / np.pi
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size_w, size_h]
            # cv2.rectangle(img,(x1, y1), (x2, y2),(255,0,0))
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(img, (x1 - left + x, y1 - bottom + y), 1, (0, 255, 255), 1)
            plot_pose_cube(img, poses[0], poses[1], poses[2], tdx=pts['nose'][0], tdy=pts['nose'][1],
                       size=(x2 - x1) // 2)
        cv2.imshow('0', img)
        cv2.waitKey(0)
        # if cv2.waitKey(0) == 27:
        #     break



def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument(
        '--model_path',
        default="./models/pretrained/checkpoint_epoch_final.pth",
        type=str)
    parser.add_argument(
        '--images_path',
        default="/home/unaguo/hanson/data/landmark/WFLW191104/WFLW_images/8--Election_Campain",
        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)