import scipy.io as sio
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys
import cv2
import os
# from moviepy.editor import *
import numpy as np
import argparse
from pfld.utils import plot_pose_cube
import random

def get_args():
	parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
	                                             "and creates database for training.",
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--db", type=str, default='AFLW2000',
	                    help="path to database")
	parser.add_argument("--root", type=str, default='/media/data4T1/hanson/Landmarks/300W_LP/',
	                    help="path to database")
	parser.add_argument("--output", type=str, default='/media/data4T1/hanson/Landmarks/300W_LP/output',
	                    help="path to output file")
	parser.add_argument("--img_size", type=int, default=112,
	                    help="output image size")
	parser.add_argument("--is_show", type=bool, default=False,
	                    help="output image size")
	args = parser.parse_args()
	return args


def pts_68_index_table():
	# 300WLP <==> WFLW
	pts_98_index = {str(i): i for i in range(0, 98)}
	pts_68_index = {str(i): i for i in range(0, 68)}
	for i in range(0, 16 + 1):  # 面部轮廓
		pts_68_index[str(i)] = pts_98_index[str(i * 2)]

	for i in range(17, 21 + 1):  # 左眼眉毛
		pts_68_index[str(i)] = pts_98_index[str(i - 17 + 33)]

	for i in range(22, 26 + 1):  # 右眼眉毛
		pts_68_index[str(i)] = pts_98_index[str(i - 22 + 42)]

	for i in range(27, 35 + 1):  # 鼻子
		pts_68_index[str(i)] = pts_98_index[str(i - 27 + 51)]

	# 左眼
	pts_68_index[str(36)] = pts_98_index[str(60)]
	pts_68_index[str(37)] = pts_98_index[str(61)]
	pts_68_index[str(38)] = pts_98_index[str(63)]
	pts_68_index[str(39)] = pts_98_index[str(64)]
	pts_68_index[str(40)] = pts_98_index[str(65)]
	pts_68_index[str(41)] = pts_98_index[str(67)]

	# 右眼
	pts_68_index[str(42)] = pts_98_index[str(68)]
	pts_68_index[str(43)] = pts_98_index[str(69)]
	pts_68_index[str(44)] = pts_98_index[str(71)]
	pts_68_index[str(45)] = pts_98_index[str(72)]
	pts_68_index[str(46)] = pts_98_index[str(73)]
	pts_68_index[str(47)] = pts_98_index[str(75)]

	for i in range(48, 67 + 1):  # 嘴巴
		pts_68_index[str(i)] = pts_98_index[str(i - 48 + 76)]
	return pts_68_index


def main():
	args = get_args()
	img_size = args.img_size
	pts_68_index = pts_68_index_table()

	output_db_root = os.path.join(args.output, args.db)
	if not os.path.exists(output_db_root):
		os.makedirs(output_db_root)

	onlyfiles_mat = [f for f in listdir(join(args.root ,args.db)) if isfile(join(args.root ,args.db, f)) and join(args.root ,args.db, f).endswith('.mat')]
	onlyfiles_jpg = [f for f in listdir(join(args.root ,args.db)) if isfile(join(args.root ,args.db, f)) and join(args.root ,args.db, f).endswith('.jpg')]
	# AFW_134212_1_0_pts
	onlyfiles_mat.sort()
	onlyfiles_jpg.sort()
	print(len(onlyfiles_jpg))
	print(len(onlyfiles_mat))
	out_imgs = []
	out_poses = []
	with open(os.path.join(args.output, args.db + '_label.txt'),'w') as fw:
		for i in tqdm(range(len(onlyfiles_jpg))):


			img_name = onlyfiles_jpg[i]
			mat_name = onlyfiles_mat[i]

			save_path = args.db + '/' + img_name

			# print(img_name)
			# print(lds_name)
			# print(mat_name)
			img_name_split = img_name.split('.')
			mat_name_split = mat_name.split('.')

			if img_name_split[0] != mat_name_split[0]:
				print('Mismatched!')
				sys.exit()

			mat_contents = sio.loadmat(os.path.join(args.root, args.db, mat_name))
			img = cv2.imread(os.path.join(args.root, args.db, img_name))
			pose_para = mat_contents['Pose_Para'][0]
			pt2d = mat_contents['pt2d']
			pt2d_x = pt2d[0,:]
			pt2d_y = pt2d[1,:]
			# I found negative value in AFLW2000. It need to be removed.
			pt2d_idx = pt2d_x > 0.0
			pt2d_idy = pt2d_y > 0.0

			pt2d_id = pt2d_idx
			if sum(pt2d_idx) > sum(pt2d_idy):
				pt2d_id = pt2d_idy

			pt2d_x = pt2d_x[pt2d_id]
			pt2d_y = pt2d_y[pt2d_id]

			img_h = img.shape[0]
			img_w = img.shape[1]

			# Crop the face loosely
			x_min = int(min(pt2d_x))
			x_max = int(max(pt2d_x))
			y_min = int(min(pt2d_y))
			y_max = int(max(pt2d_y))

			h = y_max-y_min
			w = x_max-x_min

			# ad = 0.4
			ad = random.uniform(0.12, 0.3)
			x_min = max(int(x_min - ad * w), 0)
			x_max = min(int(x_max + ad * w), img_w - 1)
			y_min = max(int(y_min - ad * h), 0)
			y_max = min(int(y_max + ad * h), img_h - 1)


			img = img[y_min:y_max,x_min:x_max]

			# Checking the cropped image

			w_factor = img_size / float(img.shape[1])
			h_factor = img_size / float(img.shape[0])
			img = cv2.resize(img, (img_size, img_size))
			save_img = img.copy()
			landmarks98 = np.array([-256.0 for i in range(98 * 2)])
			for k in range(pt2d_x.shape[0]):
				center = (int((pt2d_x[k]-x_min) * w_factor), int((pt2d_y[k]- y_min) * h_factor))
				cv2.circle(img,center ,1,(255,255,0),1)

			pitch = pose_para[0] * 180 / np.pi
			yaw = pose_para[1] * 180 / np.pi
			roll = pose_para[2] * 180 / np.pi

			euler_angles = np.array([yaw, pitch, roll])
			plot_pose_cube(img, yaw, pitch, roll, tdx=50, tdy=50,
						   size=50)

			landmark_str = ' '.join(list(map(str, landmarks98.reshape(-1).tolist())))
			euler_angles_str = ' '.join(list(map(str, euler_angles.reshape(-1).tolist())))
			label = '{} {} 0 0 0 0 0 0 {}\n'.format(save_path, landmark_str, euler_angles_str)
			fw.write(label)
			cv2.imwrite(os.path.join(output_db_root, img_name), save_img)
			if args.is_show:
				cv2.imshow('check',img)
				cv2.waitKey(0)
			# out_imgs.append(img)
			# out_poses.append(cont_labels)

if __name__ == '__main__':
	main()








