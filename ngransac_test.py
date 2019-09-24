import numpy as np
import cv2
import random
import os

import torch
import ngransac
import time

from network import CNNet
from dataset import SparseDataset
import util

parser = util.create_parser(
	description = "Test NG-RANSAC on pre-calculated correspondences.")

parser.add_argument('--dataset', '-ds', default='reichstag', 
	help='which dataset to use')

parser.add_argument('--batchmode', '-bm', action='store_true',
	help='loop over all test datasets defined in util.py')

parser.add_argument('--variant', '-v', default='test',	
	help='subfolder of the dataset to use')

parser.add_argument('--hyps', '-hyps', type=int, default=1000, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--evalbinsize', '-eb', type=float, default=5, 
	help='bin size when calculating the AUC evaluation score, 5 was used by Yi et al., and therefore also in the NG-RANSAC paper for reasons of comparability; for accurate AUC values, set to e.g. 0.1')

parser.add_argument('--model', '-m', default='',
	help='model to load, leave empty and the script infers an appropriate pre-trained model from the other settings')

parser.add_argument('--uniform', '-u', action='store_true', 
	help='use uniform probabilities intead of predicted probabilities (RANSAC instead of NG-RANSAC)')

parser.add_argument('--refine', '-ref', action='store_true', 
	help='refine using the 8point algorithm on all inliers, only used for fundamental matrix estimation (-fmat)')

opt = parser.parse_args()

print("")


if opt.uniform:
	print("Using uniform sampling (no model loaded).")
else:
	# load a model, either directly provided by the user, or infer a pre-trained model from the command line parameters
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

# construct folder that should contain pre-calculated correspondences
data_folder = opt.variant + '_data'
if opt.orb:
	data_folder += '_orb'
if opt.rootsift:
	data_folder += '_rs'

# collect datasets to be used for testing
if opt.batchmode:
	datasets = util.test_datasets
	print("\n=== BATCH MODE: Doing evaluation for", len(datasets), "datasets. =================")
else:
	datasets = [opt.dataset]

# loop over datasets, perform a separate evaluation per dataset
for dataset in datasets:
	
	print('Starting evaluation for dataset:', dataset, data_folder, "\n")

	testset = SparseDataset(['traindata/' + dataset + '/'+data_folder+'/'], opt.ratio, opt.nfeatures, opt.fmat, opt.nosideinfo)
	testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6, batch_size=opt.batchsize)

	avg_model_time = 0 # runtime of the network forward pass
	avg_ransac_time = 0 # runtime of RANSAC
	avg_counter = 0

	# essential matrix evaluation
	pose_losses = []

	# evaluation according to "deep fundamental matrix" (Ranftl and Koltun, ECCV 2018)
	avg_F1 = 0
	avg_inliers = 0
	epi_errors = []
	invalid_pairs = 0

	with torch.no_grad():
		for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in testset_loader:

			print("Processing batch", avg_counter+1, "of", len(testset_loader))

			gt_R = gt_R.numpy()
			gt_t = gt_t.numpy()

			start_time = time.time()

			if opt.uniform:
				# unfiform sampling intead of neural guidance (vanilla RANSAC)
				probs = torch.ones((correspondences.size(0), 1, correspondences.size(2), 1))
			else:
				# predict sampling weights /neural guidance
				log_probs = model(correspondences.cuda())
				probs = torch.exp(log_probs).cpu()

			avg_model_time += (time.time()-start_time) / opt.batchsize
			ransac_time = 0

			# loop over batch
			for b in range(correspondences.size(0)):

				gradients = torch.zeros(probs[b].size()) # not used in test mode, indicates which correspondence have been sampled
				inliers = torch.zeros(probs[b].size()) # inlier mask of winning model
				rand_seed = random.randint(0, 10000) # provide a random seed for C++

				if opt.fmat: 

					# === CASE FUNDAMENTAL MATRIX =========================================

					# restore pixel coordinates
					util.denormalize_pts(correspondences[b, 0:2], im_size1[b])
					util.denormalize_pts(correspondences[b, 2:4], im_size2[b])

					F = torch.zeros((3, 3))

					#run NG-RANSAC
					start_time = time.time()
					incount = ngransac.find_fundamental_mat(correspondences[b], probs[b], rand_seed, opt.hyps, opt.threshold, opt.refine, F, inliers, gradients)
					ransac_time += time.time()-start_time
					
					# essential matrix from fundamental matrix (for evaluation via relative pose)
					E = K2[b].transpose(0, 1).mm(F.mm(K1[b]))

					pts1 = correspondences[b,0:2].numpy()
					pts2 = correspondences[b,2:4].numpy()

					# evaluation of F matrix via correspondences
					valid, F1, epi_inliers, epi_error = util.f_error(pts1, pts2, F.numpy(), gt_F[b].numpy(), opt.threshold)

					if valid:
						avg_F1 += F1
						avg_inliers += epi_inliers
						epi_errors.append(epi_error)
					else:
						# F matrix evlaution failed (ground truth model had no inliers)
						invalid_pairs += 1	

					# normalize correspondences using the calibration parameters for the calculation of pose errors
					pts1 = cv2.undistortPoints(pts1.transpose(2, 1, 0), K1[b].numpy(), None)
					pts2 = cv2.undistortPoints(pts2.transpose(2, 1, 0), K2[b].numpy(), None)				

				else: 

					# === CASE ESSENTIAL MATRIX =========================================

					E = torch.zeros((3, 3)).float()
					
					#run NG-RANSAC
					start_time = time.time()
					incount = ngransac.find_essential_mat(correspondences[b], probs[b], rand_seed, opt.hyps, opt.threshold, E, inliers, gradients)		
					ransac_time += time.time()-start_time

					pts1 = correspondences[b,0:2].squeeze().numpy().T
					pts2 = correspondences[b,2:4].squeeze().numpy().T				

				inliers = inliers.byte().numpy().ravel()
				E = E.double().numpy()
				K = np.eye(3)
				R = np.eye(3)
				t = np.zeros((3,1))

				# evaluation of relative pose (essential matrix)
				cv2.recoverPose(E, pts1, pts2, K, R, t, inliers)

				dR, dT = util.pose_error(R, gt_R[b], t, gt_t[b])
				pose_losses.append(max(float(dR), float(dT)))

			avg_ransac_time += ransac_time /  opt.batchsize
			avg_counter += 1

	print("\nAvg. Model Time: %dms" % (avg_model_time / avg_counter*1000))
	print("Avg. RANSAC Time: %dms" % (avg_ransac_time / avg_counter*1000))
	 
	# calculate AUC of pose losses
	thresholds = [5, 10, 20]
	AUC = util.AUC(losses = pose_losses, thresholds = thresholds, binsize = opt.evalbinsize)

	print("\n=== Relative Pose Accuracy ===========================")
	print("AUC for %ddeg/%ddeg/%ddeg: %.2f/%.2f/%.2f\n" % (thresholds[0], thresholds[1], thresholds[2], AUC[0], AUC[1], AUC[2]))

	if opt.fmat:

		print("\n=== F-Matrix Evaluation ==============================")

		if len(epi_errors) == 0:
			print("F-Matrix evaluation failed because no ground truth inliers were found.")
			print("Check inlier threshold?.")
		else:
			avg_F1 /= len(epi_errors)
			avg_inliers /= len(epi_errors)

			epi_errors.sort()
			mean_epi_err = sum(epi_errors) / len(epi_errors)
			median_epi_err = epi_errors[int(len(epi_errors)/2)]

			print("Invalid Pairs (ignored in the following metrics):", invalid_pairs)
			print("F1 Score: %.2f%%" % (avg_F1 * 100))
			print("%% Inliers: %.2f%%" % (avg_inliers * 100))
			print("Mean Epi Error: %.2f" % mean_epi_err)
			print("Median Epi Error: %.2f" % median_epi_err)

	session_string = util.create_session_string('test', opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session)

	# write evaluation results to file
	out_dir = 'results/' + dataset + '/'
	if not os.path.isdir(out_dir): os.makedirs(out_dir)

	with open(out_dir + '%s.txt' % (session_string), 'w', 1) as f:
		f.write('%f %f %f' % (AUC[0], AUC[1], AUC[2]))
		if opt.fmat and len(epi_errors) > 0: f.write(' %f %f %f %f' % (avg_F1, avg_inliers, mean_epi_err, median_epi_err))
		f.write('\n')
