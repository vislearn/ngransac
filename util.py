import numpy as np
import cv2
import math
import argparse

def normalize_pts(pts, im_size):
	"""Normalize image coordinate using the image size.

	Pre-processing of correspondences before passing them to the network to be 
	independent of image resolution.
	Re-scales points such that max image dimension goes from -0.5 to 0.5.
	In-place operation.

	Keyword arguments:
	pts -- 3-dim array conainting x and y coordinates in the last dimension, first dimension should have size 1.
	im_size -- image height and width
	"""	

	pts[0, :, 0] -= float(im_size[1]) / 2
	pts[0, :, 1] -= float(im_size[0]) / 2
	pts /= float(max(im_size))

def denormalize_pts(pts, im_size):
	"""Undo image coordinate normalization using the image size.

	In-place operation.

	Keyword arguments:
	pts -- N-dim array conainting x and y coordinates in the first dimension
	im_size -- image height and width
	"""	
	pts *= max(im_size)
	pts[0] += im_size[1] / 2
	pts[1] += im_size[0] / 2

def AUC(losses, thresholds, binsize):
	"""Compute the AUC up to a set of error thresholds.

	Return mutliple AUC corresponding to multiple threshold provided.

	Keyword arguments:
	losses -- list of losses which the AUC should be calculated for
	thresholds -- list of threshold values up to which the AUC should be calculated
	binsize -- bin size to be used fo the cumulative histogram when calculating the AUC, the finer the more accurate
	"""

	bin_num = int(max(thresholds) / binsize)
	bins = np.arange(bin_num + 1) * binsize  

	hist, _ = np.histogram(losses, bins) # histogram up to the max threshold
	hist = hist.astype(np.float32) / len(losses) # normalized histogram
	hist = np.cumsum(hist) # cumulative normalized histogram
	 
	# calculate AUC for each threshold
	return [np.mean(hist[:int(t / binsize)]) for t in thresholds]


def pose_error(R, gt_R, t, gt_t):
	"""Compute the angular error between two rotation matrices and two translation vectors.


	Keyword arguments:
	R -- 2D numpy array containing an estimated rotation
	gt_R -- 2D numpy array containing the corresponding ground truth rotation
	t -- 2D numpy array containing an estimated translation as column
	gt_t -- 2D numpy array containing the corresponding ground truth translation
	"""

	# calculate angle between provided rotations
	dR = np.matmul(R, np.transpose(gt_R))
	dR = cv2.Rodrigues(dR)[0]
	dR = np.linalg.norm(dR) * 180 / math.pi
	
	# calculate angle between provided translations
	dT = float(np.dot(gt_t.T, t))
	dT /= float(np.linalg.norm(gt_t))

	if dT > 1 or dT < -1:
		print("Domain warning! dT:", dT)
		dT = max(-1, min(1, dT))
	dT = math.acos(dT) * 180 / math.pi

	return dR, dT

def f_error(pts1, pts2, F, gt_F, threshold):
	"""Compute multiple evaluaton measures for a fundamental matrix.

	Return (False, 0, 0, 0) if the evaluation fails due to not finding inliers for the ground truth model, 
	else return() True, F1 score, % inliers, mean epipolar error of inliers).

	Follows the evaluation procedure in:
	"Deep Fundamental Matrix Estimation"
	Ranftl and Koltun
	ECCV 2018

	Keyword arguments:
	pts1 -- 3D numpy array containing the feature coordinates in image 1, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
	pts2 -- 3D numpy array containing the feature coordinates in image 2, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
	F -- 2D numpy array containing an estimated fundamental matrix
	gt_F -- 2D numpy array containing the corresponding ground truth fundamental matrix
	threshold -- inlier threshold for the epipolar error in pixels
	"""

	EPS = 0.00000000001
	num_pts = pts1.shape[1]

	# 2D coordinates to 3D homogeneous coordinates
	hom_pts1 = np.concatenate((pts1[:,:,0], np.ones((1, num_pts))), axis=0)
	hom_pts2 = np.concatenate((pts2[:,:,0], np.ones((1, num_pts))), axis=0)

	def epipolar_error(hom_pts1, hom_pts2, F):
		"""Compute the symmetric epipolar error."""
		res  = 1 / np.linalg.norm(F.T.dot(hom_pts2)[0:2], axis=0)
		res += 1 / np.linalg.norm(F.dot(hom_pts1)[0:2], axis=0)
		res *= abs(np.sum(hom_pts2 * np.matmul(F, hom_pts1), axis=0))
		return res

	# determine inliers based on the epipolar error
	est_res = epipolar_error(hom_pts1, hom_pts2, F)
	gt_res = epipolar_error(hom_pts1, hom_pts2, gt_F)

	est_inliers = (est_res < threshold)
	gt_inliers = (gt_res < threshold)
	true_positives = est_inliers & gt_inliers

	gt_inliers = float(gt_inliers.sum())

	if gt_inliers > 0:

		est_inliers = float(est_inliers.sum())				
		true_positives = float(true_positives.sum())

		precision = true_positives / (est_inliers + EPS)
		recall = true_positives / (gt_inliers + EPS)

		F1 = 2 * precision * recall / (precision + recall + EPS)
		inliers = est_inliers / num_pts

		epi_mask = (gt_res < 1)
		if epi_mask.sum() > 0:
			epi_error = float(est_res[epi_mask].mean())
		else:
			# no ground truth inliers for the fixed 1px threshold used for epipolar errors
			return False, 0, 0, 0

		return True, F1, inliers, epi_error		
	else:
		# no ground truth inliers for the user provided threshold
		return False, 0, 0, 0

def rootSift(desc):
	"""Apply root sift normalization to a given set of descriptors.

	See details in:
	"Three Things Everyone Should Know to Improve Object Retrieval"
	Arandjelovic and Zisserman
	CVPR 2012

	Keyword arguments:
	desc -- 2D numpy array containing the descriptors in its rows
	"""

	desc_norm = np.linalg.norm(desc, ord=1, axis=1)
	desc_norm += 1 # avoid division by zero
	desc_norm = np.expand_dims(desc_norm, axis=1)
	desc_norm = np.repeat(desc_norm, desc.shape[1], axis=1)

	desc = np.divide(desc, desc_norm)
	return np.sqrt(desc)


def create_parser(description):
	"""Create a default command line parser with the most common options.

	Keyword arguments:
	description -- description of the main functionality of a script/program
	"""

	parser = argparse.ArgumentParser(
		description=description,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--fmat', '-fmat', action='store_true', 
		help='estimate the fundamental matrix, instead of the essential matrix')

	parser.add_argument('--rootsift', '-rs', action='store_true', 
		help='use RootSIFT normalization')

	parser.add_argument('--orb', '-orb', action='store_true', 
		help='use ORB instead of SIFT')

	parser.add_argument('--nfeatures', '-nf', type=int, default=2000, 
		help='fixes number of features by clamping/replicating, set to -1 for dynamic feature count but then batchsize (-bs) has to be set to 1')
	
	parser.add_argument('--ratio', '-r', type=float, default=1.0, 
		help='apply Lowes ratio filter with the given ratio threshold, 1.0 does nothing')

	parser.add_argument('--nosideinfo', '-nos', action='store_true', 
		help='Do not provide side information (matching ratios) to the network. The network should be trained and tested consistently.')

	parser.add_argument('--threshold', '-t', type=float, default=0.001, 
		help='inlier threshold. Recommended values are 0.001 for E matrix estimation, and 0.1 or 1.0 for F matrix estimation')

	parser.add_argument('--resblocks', '-rb', type=int, default=12, 
		help='number of res blocks of the network')

	parser.add_argument('--batchsize', '-bs', type=int, default=32, help='batch size')

	parser.add_argument('--session', '-sid', default='',
		help='custom session name appended to output files, useful to separate different runs of a script')

	return parser

def create_session_string(prefix, fmat, orb, rootsift, ratio, session):
	"""Create an identifier string from the most common parameter options.

	Keyword arguments:
	prefix -- custom string appended at the beginning of the session string
	fmat -- bool indicating whether fundamental matrices or essential matrices are estimated
	orb -- bool indicating whether ORB features or SIFT features are used
	rootsift -- bool indicating whether RootSIFT normalization is used
	ratio -- threshold for Lowe's ratio filter
	session -- custom string appended at the end of the session string
	"""
	session_string = prefix + '_'
	if fmat:
		session_string += 'F_'
	else:
		session_string += 'E_'
	
	if orb:	session_string += 'orb_'
	if rootsift: session_string += 'rs_'
	session_string += 'r%.2f_' % ratio
	session_string += session

	return session_string

#list of all test datasets used in the NG-RANSAC paper
outdoor_test_datasets = [
'buckingham_palace',
'notre_dame_front_facade',
'sacre_coeur',
'reichstag',
'fountain',
'herzjesu',
]

indoor_test_datasets = [
'brown_cogsci_2---brown_cogsci_2---skip-10-dilate-25',
'brown_cogsci_6---brown_cogsci_6---skip-10-dilate-25',
'brown_cogsci_8---brown_cogsci_8---skip-10-dilate-25',
'brown_cs_3---brown_cs3---skip-10-dilate-25',
'brown_cs_7---brown_cs7---skip-10-dilate-25',
'harvard_c4---hv_c4_1---skip-10-dilate-25',
'harvard_c10---hv_c10_2---skip-10-dilate-25',
'harvard_corridor_lounge---hv_lounge1_2---skip-10-dilate-25',
'harvard_robotics_lab---hv_s1_2---skip-10-dilate-25',
'hotel_florence_jx---florence_hotel_stair_room_all---skip-10-dilate-25',
'mit_32_g725---g725_1---skip-10-dilate-25',
'mit_46_6conf---bcs_floor6_conf_1---skip-10-dilate-25',
'mit_46_6lounge---bcs_floor6_long---skip-10-dilate-25',
'mit_w85g---g_0---skip-10-dilate-25',
'mit_w85h---h2_1---skip-10-dilate-25',
]

test_datasets = outdoor_test_datasets + indoor_test_datasets
