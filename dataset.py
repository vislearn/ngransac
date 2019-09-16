import numpy as np
import torch
import os
import cv2
import math
import util

from torch.utils.data import Dataset

class SparseDataset(Dataset):
	"""Sparse correspondences dataset."""

	def __init__(self, folders, ratiothreshold, nfeatures, fmat=False, overwrite_side_info=False):

		self.nfeatures = nfeatures # ensure fixed number of features, -1 keeps original feature count
		self.ratiothreshold = ratiothreshold # threshold for Lowe's ratio filter
		self.overwrite_side_info = overwrite_side_info # if true, provide no side information to the neural guidance network
		
		# collect precalculated correspondences of all provided datasets
		self.files = []
		for folder in folders:
			self.files += [folder + f for f in os.listdir(folder)]

		self.fmat = fmat # estimate fundamental matrix instead of essential matrix
		self.minset = 5 # minimal set size for essential matrices
		if fmat: self.minset = 7 # minimal set size for fundamental matrices
			
	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):

		# load precalculated correspondences
		data = np.load(self.files[idx], allow_pickle=True)

		# correspondence coordinates and matching ratios (side information)
		pts1, pts2, ratios = data[0], data[1], data[2]
		# image sizes
		im_size1, im_size2 = torch.from_numpy(np.asarray(data[3])), torch.from_numpy(np.asarray(data[4]))
		# image calibration parameters
		K1, K2 = torch.from_numpy(data[5]), torch.from_numpy(data[6])
		# ground truth pose
		gt_R, gt_t = torch.from_numpy(data[7]), torch.from_numpy(data[8])

		# applying Lowes ratio criterion
		ratio_filter = ratios[0,:,0] < self.ratiothreshold

		if ratio_filter.sum() < self.minset: # ensure a minimum count of correspondences
			print("WARNING! Ratio filter too strict. Only %d correspondences would be left, so I skip it." % int(ratio_filter.sum()))
		else:
			pts1 = pts1[:,ratio_filter,:]
			pts2 = pts2[:,ratio_filter,:]
			ratios = ratios[:,ratio_filter,:]
		
		if self.overwrite_side_info:
			ratios = np.zeros(ratios.shape, dtype=np.float32)

		if self.fmat:
			# for fundamental matrices, normalize image coordinates using the image size (network should be independent to resolution)
			util.normalize_pts(pts1, im_size1)
			util.normalize_pts(pts2, im_size2)
		else:
			#for essential matrices, normalize image coordinate using the calibration parameters
			pts1 = cv2.undistortPoints(pts1, K1.numpy(), None)
			pts2 = cv2.undistortPoints(pts2, K2.numpy(), None)

		# stack image coordinates and side information into one tensor
		correspondences = np.concatenate((pts1, pts2, ratios), axis=2)
		correspondences = np.transpose(correspondences)
		correspondences = torch.from_numpy(correspondences)

		if self.nfeatures > 0:
			# ensure that there are exactly nfeatures entries in the data tensor 
			if correspondences.size(1) > self.nfeatures:
				rnd = torch.randperm(correspondences.size(1))
				correspondences = correspondences[:,rnd,:]
				correspondences = correspondences[:,0:self.nfeatures]

			if correspondences.size(1) < self.nfeatures:
				result = correspondences
				for i in range(0, math.ceil(self.nfeatures / correspondences.size(1) - 1)):
					rnd = torch.randperm(correspondences.size(1))
					result = torch.cat((result, correspondences[:,rnd,:]), dim=1)
				correspondences = result[:,0:self.nfeatures]

		# construct the ground truth essential matrix from the ground truth relative pose
		gt_E = torch.zeros((3,3))
		gt_E[0, 1] = -float(gt_t[2,0])
		gt_E[0, 2] = float(gt_t[1,0])
		gt_E[1, 0] = float(gt_t[2,0])
		gt_E[1, 2] = -float(gt_t[0,0])
		gt_E[2, 0] = -float(gt_t[1,0])
		gt_E[2, 1] = float(gt_t[0,0])

		gt_E = gt_E.mm(gt_R)

		# fundamental matrix from essential matrix
		gt_F = K2.inverse().transpose(0, 1).mm(gt_E).mm(K1.inverse())

		return correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2


