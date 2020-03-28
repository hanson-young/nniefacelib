# coding=utf-8
import argparse
import os
import time
from math import ceil
import sys
sys.path.append('/home/unaguo/hanson/caffe-ssdv1.0/python')
import caffe
import cv2
import numpy as np
sys.path.append('../pfld')
from utils import plot_pose_cube
from hdface.hdface import hdface_detector


parser = argparse.ArgumentParser()
parser.add_argument('--caffe_prototxt_path', default="models/pfpld.prototxt", type=str, help='caffe_prototxt_path')
parser.add_argument('--caffe_model_path', default="models/pfpld.caffemodel", type=str, help='caffe_model_path')
parser.add_argument('--input_size', default="112,112", type=str, help='define network input size,format: width,height')
parser.add_argument('--imgs_path', default="/home/unaguo/hanson/data/faces-detector/data/WFLW/test_data/WFLW", type=str, help='imgs dir')
parser.add_argument('--results_path', default="./result", type=str, help='results dir')
parser.add_argument('--mode', default="cpu", type=str, help='cpu or gpu')
args = parser.parse_args()

if args.mode == "cpu":
    caffe.set_mode_cpu()
elif args.mode == "gpu":
    caffe.set_mode_gpu()
image_mean = np.array([0, 0, 0])
image_std = 255.0
det = hdface_detector(use_cuda=True)
def inference():
    net = caffe.Net(args.caffe_prototxt_path, args.caffe_model_path, caffe.TEST)
    input_size = [int(v.strip()) for v in args.input_size.split(",")]
    witdh = input_size[0]
    height = input_size[1]
    # priors = define_img_size(input_size)
    net.blobs['input'].reshape(1, 3, height, witdh)
    result_path = args.results_path
    imgs_path = args.imgs_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    listdir = os.listdir(imgs_path)

    for file_path in listdir:
        img_path = os.path.join(imgs_path, file_path)
        img_ori = cv2.imread(img_path)
        img_det = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        result = det.detect_face(img_det)
        img_h, img_w = img_ori.shape[:2]
        for i in range(len(result)):
            box = result[i]['box']
            cls = result[i]['cls']
            pts = result[i]['pts']
            x1, y1, x2, y2 = box
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), (255, 255, 25))
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
            if x1 >= img_w:
                right = x2 - img_w
            if y1 >= img_h:
                bottom = y2 - img_h

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            cropped = img_ori[y1:y2, x1:x2]
            print(top, bottom, left, right)
            cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

            tmp_batch = np.zeros([1, 3, height, witdh], dtype=np.float32)
            cropped = cv2.resize(cropped, (witdh, height))
            image = (cropped - image_mean) / image_std
            tmp_batch[0, :, :, :] = image.transpose(2, 0, 1)
            net.blobs['input'].data[...] = tmp_batch
            time_time = time.time()
            res = net.forward()
            poses = res['pose'][0]
            landms = res['landms'][0]
            poses = poses * 180 / np.pi
            landms = landms
            print("inference time: {} s".format(round(time.time() - time_time, 4)))
            for i in range(98):
                lx, ly = (int(landms[i * 2] * size_w + x1 - left), int(landms[i * 2 + 1] * size_h + y1 - bottom))
                cv2.circle(img_ori, (lx, ly),1,(0,255,255),2)
            plot_pose_cube(img_ori,poses[0], poses[1],poses[2],tdx=pts['nose'][0], tdy=pts['nose'][1], size=(x2 - x1) // 2)
            cv2.imwrite(os.path.join(result_path, file_path), img_ori)
            # print("result_pic is written to {}".format(os.path.join(result_path, file_path)))
            # cv2.imshow("show", img_ori)
            # cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    inference()
