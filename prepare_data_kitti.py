import numpy as np
import cv2 as cv
import h5py
import math
import argparse
import os
import random

import torch
import util

# parse command line arguments
parser = argparse.ArgumentParser(
	description='Train large scale camera localization.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', '-ds', default='dataset_kitti',
	help='Root folder of the Kitti odometry dataset. Should contain folders "poses" and "sequences".')

parser.add_argument('--variant', '-v', default='train', choices=['train', 'test']
	help='Defines subfolders of the dataset ot use (split according to "Deep Fundamental Matrix", Ranftl and Koltun, ECCV 2018).')

parser.add_argument('--orb', '-orb', action='store_true', 
	help='Use ORB instead of SIFT')

parser.add_argument('--rootsift', '-rs', action='store_true', 
	help='Use rootSIFT normalization')

parser.add_argument('--nfeatures', '-nf', type=int, default=-1, 
	help='number of features per image, -1 does not restrict feature count')

opt = parser.parse_args()

if opt.variant == 'train':
	datasets = ['00','01','02','03','04','05']
else:
	datasets = ['06','07','08','09','10']

print('Using dataset: ', opt.dataset, opt.variant)

# output folder that stores pre-calculated correspondence vectors as PyTorch tensors
out_dir = 'traindata/kitti/' + opt.variant + '_data'

# depending on the settings the data folder is maked with "rs" for rootsift, and/or "08"
if opt.orb:
	out_dir += '_orb'
if opt.rootsift:
	out_dir += '_rs'
out_dir += '/'
if not os.path.isdir(out_dir): os.makedirs(out_dir)

cal_db = {} # global list of calibration and ground truth poses
img_db = {} # global list of image files
vis_pairs = [] # list of image pairs
offset = 0 # keep track of global index, combining individual sequences

for dataset in datasets:

	#images
	data_dir = opt.dataset + '/sequences/' + dataset + '/image_0/'
	#camera calibration
	cal_file = opt.dataset + '/sequences/' + dataset + '/calib.txt'
	#ground truth poses
	pose_db  = opt.dataset + '/poses/' + dataset + '.txt'

	cal_file = open(cal_file, 'r')
	pose_db = open(pose_db, 'r')

	img_files = os.listdir(data_dir)
	img_files.sort()

	calibration = cal_file.readlines()
	poses = pose_db.readlines()

	cal_file.close()
	pose_db.close()

	# calibration matrix is constant per sequence
	calibration = [float(item) for item in calibration[0].split()[1:]]
	calibration = np.array(calibration).reshape((3,4))
	calibration = calibration[0:3,0:3]

	for i, pose in enumerate(poses):

		pose = [float(p) for p in pose.split()]
		pose += [0, 0, 0, 1]

		pose = np.array(pose).reshape((4,4))
		pose = np.linalg.inv(pose)

		K = calibration
		R = pose[0:3,0:3]
		T = pose[0:3,3].reshape(1,3)
		cal_db[i+offset] = (K, R, T) # store ground truth information
		img_db[i+offset] = data_dir + img_files[i] # store image file

	# add image pairs for this sequence
	# we combine each image with the next image in the sequence
	for i in range(0, len(img_files)):
		for o in range(1, 2):
			if i+o < len(img_files):
				vis_pairs.append((i+offset, i+o+offset))

	offset = len(vis_pairs) # update global index

# setup detector
if opt.orb:
	if opt.nfeatures > 0:
		detector = cv.ORB_create(nfeatures=opt.nfeatures)
	else:
		detector = cv.ORB_create()
else:
	if opt.nfeatures > 0:
		detector = cv.xfeatures2d.SIFT_create(nfeatures=opt.nfeatures, contrastThreshold=1e-5)
	else:
		detector = cv.xfeatures2d.SIFT_create()

# randomize ordering of image pairs
random.shuffle(vis_pairs)

for i, vis_pair in enumerate(vis_pairs):

	img1_idx = vis_pair[0]
	img2_idx = vis_pair[1]

	print("\nProcessing pair %d of %d. (%d, %d)" % (i, len(vis_pairs), img1_idx, img2_idx))

	# read images
	img1 = cv.imread(img_db[img1_idx])
	img2 = cv.imread(img_db[img2_idx])

	# detect features
	kp1, desc1 = detector.detectAndCompute(img1, None)
	kp2, desc2 = detector.detectAndCompute(img2, None)

	print("Features found:", len(kp1), len(kp2))
	if min(len(kp1), len(kp2)) < 10: continue # ensure a minimum number of features

	# root sift normalization
	if opt.rootsift:
		desc1 = util.rootSift(desc1)
		desc2 = util.rootSift(desc2)

	# feature matching
	bf = cv.BFMatcher()
	matches = bf.knnMatch(desc1, desc2, k=2)

	pts1 = []
	pts2 = []
	
	#side infromation (matching ratios in this case)
	ratios = []

	for (m,n) in matches:
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)
		ratios.append(m.distance / n.distance)

	print("Matches:", len(matches))

	pts1 = np.array([pts1])
	pts2 = np.array([pts2])

	ratios = np.array([ratios])
	ratios = np.expand_dims(ratios, 2)

	K1 = cal_db[img1_idx][0]
	K2 = cal_db[img2_idx][0]

	# calculate ground truth relative pose from absolute poses
	GT_R1 = cal_db[img1_idx][1]
	GT_R2 = cal_db[img2_idx][1]
	GT_R_Rel = np.matmul(GT_R2, np.transpose(GT_R1))
	
	GT_t1 = cal_db[img1_idx][2]
	GT_t2 = cal_db[img2_idx][2]
	GT_t_Rel = GT_t2.T - np.matmul(GT_R_Rel, GT_t1.T)

	#save data tensor and ground truth transformation
	np.save(out_dir + 'pair_%d_%d.npy' % (img1_idx, img2_idx), [
		pts1.astype(np.float32), 
		pts2.astype(np.float32), 
		ratios.astype(np.float32), 
		img1.shape, 
		img2.shape, 
		K1.astype(np.float32), 
		K2.astype(np.float32), 
		GT_R_Rel.astype(np.float32), 
		GT_t_Rel.astype(np.float32)
		])
