import numpy as np
import cv2
import math
import argparse
import os
import random

import torch
import torch.optim as optim
import ngransac

from network import CNNet
from dataset import SparseDataset
import util

parser = util.create_parser('NG-RANSAC demo for a user defined image pair. Fits an essential matrix (default) or fundamental matrix (-fmat) using OpenCV RANSAC vs. NG-RANSAC.')

parser.add_argument('--image1', '-img1', default='images/demo1.jpg',
	help='path to image 1')

parser.add_argument('--image2', '-img2', default='images/demo2.jpg',
	help='path to image 2')

parser.add_argument('--outimg', '-out', default='demo.png',
	help='demo will store a matching image under this file name')

parser.add_argument('--focallength1', '-fl1', type=float, default=900, 
	help='focal length of image 1 (only used when fitting the essential matrix)')

parser.add_argument('--focallength2', '-fl2', type=float, default=900, 
	help='focal length of image 2 (only used when fitting the essential matrix)')

parser.add_argument('--model', '-m', default='',
	help='model to load, leave empty and the script infers an appropriate pre-trained model from the other settings')

parser.add_argument('--hyps', '-hyps', type=int, default=1000, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--refine', '-ref', action='store_true', 
	help='refine using the 8point algorithm on all inliers, only used for fundamental matrix estimation (-fmat)')

opt = parser.parse_args()

if opt.fmat:
	print("\nFitting Fundamental Matrix...\n")
else:
	print("\nFitting Essential Matrix...\n")

# setup detector
if opt.orb:
	print("Using ORB.\n")
	if opt.nfeatures > 0:
		detector = cv2.ORB_create(nfeatures=opt.nfeatures)
	else:
		detector = cv2.ORB_create()
else:
	if opt.rootsift:
		print("Using RootSIFT.\n")
	else:
		print("Using SIFT.\n")
	if opt.nfeatures > 0:
		detector = cv2.xfeatures2d.SIFT_create(nfeatures=opt.nfeatures, contrastThreshold=1e-5)
	else:
		detector = cv2.xfeatures2d.SIFT_create()

# loading neural guidence network
model_file = opt.model
if len(model_file) == 0:
	model_file = util.create_session_string('e2e', opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session)
	model_file = 'models/weights_' + model_file + '.net'
	print("No model file specified. Inferring pre-trained model from given parameters:")
	print(model_file)

model = CNNet(opt.resblocks)
model.load_state_dict(torch.load(model_file))
model = model.cuda()
model.eval()
print("Successfully loaded model.")

print("\nProcessing pair:")
print("Image 1: ", opt.image1)
print("Image 2: ", opt.image2)

# read images
img1 = cv2.imread(opt.image1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread(opt.image2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# calibration matrices of image 1 and 2, principal point assumed to be at the center
K1 = np.eye(3)
K1[0,0] = K1[1,1] = opt.focallength1
K1[0,2] = img1.shape[1] * 0.5
K1[1,2] = img1.shape[0] * 0.5

K2 = np.eye(3)
K2[0,0] = K2[1,1] = opt.focallength2
K2[0,2] = img2.shape[1] * 0.5
K2[1,2] = img2.shape[0] * 0.5

# detect features
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

print("\nFeature found in image 1:", len(kp1))
print("Feature found in image 2:", len(kp2))

# root sift normalization
if opt.rootsift:
	print("Using RootSIFT normalization.")
	desc1 = util.rootSift(desc1)
	desc2 = util.rootSift(desc2)

# feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

good_matches = []
pts1 = []
pts2 = []

#side information for the network (matching ratios in this case)
ratios = []

print("")
if opt.ratio < 1.0:
	print("Using Lowe's ratio filter with", opt.ratio)

for (m,n) in matches:
	if m.distance < opt.ratio*n.distance: # apply Lowe's ratio filter
		good_matches.append(m)
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)
		ratios.append(m.distance / n.distance)

print("Number of valid matches:", len(good_matches))

pts1 = np.array([pts1])
pts2 = np.array([pts2])

ratios = np.array([ratios])
ratios = np.expand_dims(ratios, 2)

# ------------------------------------------------
# fit fundamental or essential matrix using OPENCV
# ------------------------------------------------
if opt.fmat:

	# === CASE FUNDAMENTAL MATRIX =========================================

	ransac_model, ransac_inliers = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=opt.threshold, confidence=0.999)
else:
	# === CASE ESSENTIAL MATRIX =========================================

	# normalize key point coordinates when fitting the essential matrix
	pts1 = cv2.undistortPoints(pts1, K1, None)
	pts2 = cv2.undistortPoints(pts2, K2, None)

	K = np.eye(3)

	ransac_model, ransac_inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=opt.threshold)

print("\n=== Model found by RANSAC: ==========\n")
print(ransac_model)

print("\nRANSAC Inliers:", ransac_inliers.sum())

# ---------------------------------------------------
# fit fundamental or essential matrix using NG-RANSAC
# ---------------------------------------------------

if opt.fmat:
	# normalize x and y coordinates before passing them to the network
	# normalized by the image size
	util.normalize_pts(pts1, img1.shape)
	util.normalize_pts(pts2, img2.shape)

if opt.nosideinfo:
	# remove side information before passing it to the network
	ratios = np.zeros(ratios.shape)

# create data tensor of feature coordinates and matching ratios
correspondences = np.concatenate((pts1, pts2, ratios), axis=2)
correspondences = np.transpose(correspondences)
correspondences = torch.from_numpy(correspondences).float()

# predict neural guidance, i.e. RANSAC sampling probabilities
log_probs = model(correspondences.unsqueeze(0).cuda())[0] #zero-indexing creates and removes a dummy batch dimension
probs = torch.exp(log_probs).cpu()

out_model = torch.zeros((3, 3)).float() # estimated model
out_inliers = torch.zeros(log_probs.size()) # inlier mask of estimated model
out_gradients = torch.zeros(log_probs.size()) # gradient tensor (only used during training)
rand_seed = 0 # random seed to by used in C++

# run NG-RANSAC
if opt.fmat:

	# === CASE FUNDAMENTAL MATRIX =========================================

	# undo normalization of x and y image coordinates
	util.denormalize_pts(correspondences[0:2], img1.shape)
	util.denormalize_pts(correspondences[2:4], img2.shape)

	incount = ngransac.find_fundamental_mat(correspondences, probs, rand_seed, opt.hyps, opt.threshold, opt.refine, out_model, out_inliers, out_gradients)
else:

	# === CASE ESSENTIAL MATRIX =========================================

	incount = ngransac.find_essential_mat(correspondences, probs, rand_seed, opt.hyps, opt.threshold, out_model, out_inliers, out_gradients)

print("\n=== Model found by NG-RANSAC: =======\n")
print(out_model.numpy())

print("\nNG-RANSAC Inliers: ", int(incount))

# create a visualization of the matching, comparing results of RANSAC and NG-RANSAC
out_inliers = out_inliers.byte().numpy().ravel().tolist()
ransac_inliers = ransac_inliers.ravel().tolist()

match_img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2, matchColor=(75,180,60), matchesMask = ransac_inliers)
match_img_ngransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2, matchColor=(200,130,0), matchesMask = out_inliers)
match_img = np.concatenate((match_img_ransac, match_img_ngransac), axis = 0)

cv2.imwrite(opt.outimg, match_img)
print("\nDone. Visualization of the result stored as", opt.outimg)
