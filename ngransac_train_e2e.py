import numpy as np
import cv2
import random

import torch
import torch.optim as optim
import ngransac

from network import CNNet
from dataset import SparseDataset
import util

# parse command line arguments
parser = util.create_parser(
	description = "Train a neural guidance network end-to-end using a task loss.")

parser.add_argument('--datasets', '-ds', 
	default='brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25,st_peters_square',
	help='which datasets to use, separate multiple datasets by comma')

parser.add_argument('--variant', '-v', default='train',	
		help='subfolder of the dataset to use')

parser.add_argument('--hyps', '-hyps', type=int, default=16, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--samplecount', '-ss', type=int, default=4, 
	help='number of samples when approximating the expectation')

parser.add_argument('--learningrate', '-lr', type=float, default=0.00001, 
	help='learning rate')

parser.add_argument('--loss', '-l', choices=['pose', 'inliers', 'f1', 'epi'], default='pose', 
	help='Loss to use as a reward signal; "pose" means max of translational and rotational angle error, "inliers" maximizes the inlier count (self-supervised training), "f1" is the alignment of estimated inliers and ground truth inliers (only for fundamental matrixes, i.e. -fmat), "epi" is the mean epipolar error of inliers to ground truth epi lines (only for fundamental matrixes, i.e. -fmat)')

parser.add_argument('--epochs', '-e', type=int, default=100,
	help='number of epochs')

parser.add_argument('--model', '-m', default='', 
	help='load a model to contuinue training or leave empty to create a new model')

parser.add_argument('--refine', '-ref', action='store_true', 
	help='refine using the 8point algorithm on all inliers, only used for fundamental matrix estimation (-fmat)')

opt = parser.parse_args()

# construct folder that should contain pre-calculated correspondences
data_folder = opt.variant + '_data'
if opt.orb:
	data_folder += '_orb'
if opt.rootsift:
	data_folder += '_rs'

train_data = opt.datasets.split(',') #support multiple training datasets used jointly
train_data = ['traindata/' + ds + '/' + data_folder + '/' for ds in train_data]

print('Using datasets:')
for d in train_data:
	print(d)

trainset = SparseDataset(train_data, opt.ratio, opt.nfeatures, opt.fmat, opt.nosideinfo)
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6, batch_size=opt.batchsize)

print("\nImage pairs: ", len(trainset), "\n")

# create or load model
model = CNNet(opt.resblocks)
if len(opt.model) > 0:
	model.load_state_dict(torch.load(opt.model))
model = model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=opt.learningrate)

iteration = 0

# keep track of the training progress
session_string = util.create_session_string('e2e', opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session)
train_log = open('log_%s.txt' % (session_string), 'w', 1)

# main training loop
for epoch in range(0, opt.epochs):	

	print("=== Starting Epoch", epoch, "==================================")

	# store the network every so often
	torch.save(model.state_dict(), './weights_%s.net' % (session_string))

	# main training loop in the current epoch
	for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in trainset_loader:

		gt_R = gt_R.numpy()
		gt_t = gt_t.numpy()

		# predict neural guidance
		log_probs = model(correspondences.cuda())
		probs = torch.exp(log_probs).cpu()

		# this tensor will contain the gradients for the entire batch
		log_probs_grad = torch.zeros(log_probs.size())

		avg_loss = 0

		#loop over batch
		for b in range(correspondences.size(0)):

			# we sample multiple times per input and keep the gradients and losse in the following lists
			log_prob_grads = [] 
			losses = []

			# loop over samples for approximating the expected loss
			for s in range(opt.samplecount):

				# gradient tensor of the current sample
				# when running NG-RANSAC, this tensor will indicate which correspondences have been samples
				# this is multiplied with the loss of the sample to yield the gradients for log-probabilities
				gradients = torch.zeros(probs[b].size()) 

				 # inlier mask of the best model
				inliers = torch.zeros(probs[b].size())

				# random seed used in C++ (would be initialized in each call with the same seed if not provided from outside)
				rand_seed = random.randint(0, 10000) 

				if opt.fmat:

					# === CASE FUNDAMENTAL MATRIX =========================================

					if s == 0: #denormalization is inplace, so do it for the first sample only
						# restore pixel coordinates
						util.denormalize_pts(correspondences[b, 0:2], im_size1[b])
						util.denormalize_pts(correspondences[b, 2:4], im_size2[b])

					# run NG-RANSAC
					F = torch.zeros((3, 3))
					ngransac.find_fundamental_mat(correspondences[b], probs[b], rand_seed, opt.hyps, opt.threshold, opt.refine, F, inliers, gradients)	

					# essential matrix from fundamental matrix (for evaluation)
					E = K2[b].transpose(0, 1).mm(F.mm(K1[b]))

					pts1 = correspondences[b,0:2].numpy()
					pts2 = correspondences[b,2:4].numpy()

					# compute fundamental matrix metrics if they are used as training signal
					if opt.loss is not 'pose':
						valid, F1, incount, epi_error = util.f_error(pts1, pts2, F.numpy(), gt_F[b].numpy(), opt.threshold)
					
					# normalize correspondences using the calibration parameters for the calculation of pose errors
					pts1 = cv2.undistortPoints(pts1.transpose(2, 1, 0), K1[b].numpy(), None)
					pts2 = cv2.undistortPoints(pts2.transpose(2, 1, 0), K2[b].numpy(), None)				

				else:

					# === CASE ESSENTIAL MATRIX =========================================

					# run NG-RANSAC
					E = torch.zeros((3, 3)).float()
					incount = ngransac.find_essential_mat(correspondences[b], probs[b], rand_seed, opt.hyps, opt.threshold, E, inliers, gradients)		
					incount /= correspondences.size(2)

					pts1 = correspondences[b,0:2].squeeze().numpy().T
					pts2 = correspondences[b,2:4].squeeze().numpy().T	

				# choose the user-defined training signal
				if opt.loss == 'inliers':
					loss = -incount
				elif opt.loss == 'f1' and opt.fmat:					
					loss = -F1
				elif opt.loss == 'epi' and opt.fmat:
					loss = epi_error
				else:
					# evaluation of relative pose (essential matrix)
					inliers = inliers.byte().numpy().ravel()
					E = E.double().numpy()
					K = np.eye(3)
					R = np.eye(3)
					t = np.zeros((3,1))

					cv2.recoverPose(E, pts1, pts2, K, R, t, inliers)
					dR, dT = util.pose_error(R, gt_R[b], t, gt_t[b])
					loss = max(float(dR), float(dT))
						
				log_prob_grads.append(gradients)
				losses.append(loss)
		
			# calculate the gradients of the expected loss
			baseline = sum(losses) / len(losses) #expected loss
			for i, l in enumerate(losses): # substract baseline for each sample to reduce gradient variance
				log_probs_grad[b] += log_prob_grads[i] * (l - baseline) / opt.samplecount

			avg_loss += baseline

		avg_loss /= correspondences.size(0)

		train_log.write('%d %f\n' % (iteration, avg_loss))

		# update model
		torch.autograd.backward((log_probs), (log_probs_grad.cuda()))
		optimizer.step() 
		optimizer.zero_grad()

		print("Iteration: ", iteration, "Loss: ", avg_loss)

		iteration += 1