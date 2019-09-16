import numpy as np
import math

import torch
import torch.optim as optim
import ngransac

from network import CNNet
from dataset import SparseDataset
import util

# parse command line arguments
parser = util.create_parser(
	description = "Train a neural guidance network using correspondence distance to a ground truth model to calculate target probabilities.")

parser.add_argument('--datasets', '-ds', 
	default='brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25,st_peters_square',
	help='which datasets to use, separate multiple datasets by comma')

parser.add_argument('--variant', '-v', default='train',	
		help='subfolder of the dataset to use')

parser.add_argument('--learningrate', '-lr', type=float, default=0.001, 
	help='learning rate')

parser.add_argument('--epochs', '-e', type=int, default=1000,
	help='number of epochs')

parser.add_argument('--model', '-m', default='',
	help='load a model to contuinue training or leave empty to create a new model')

opt = parser.parse_args()

# construct folder that should contain pre-calculated correspondences
data_folder = opt.variant + '_data'
if opt.orb:
	data_folder += '_orb'
if opt.rootsift:
	data_folder += '_rs'

train_data = opt.datasets.split(',')  #support multiple training datasets used jointly
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
session_string = util.create_session_string('init', opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session)
train_log = open('log_%s.txt' % (session_string), 'w', 1)

# in the initalization we optimize the KLDiv of the predicted distribution and the target distgribution (see paper supplement A, Eq. 12)
distLoss = torch.nn.KLDivLoss(reduction='sum')

# main training loop
for epoch in range(0, opt.epochs):	

	print("=== Starting Epoch", epoch, "==================================")

	# store the network every epoch
	torch.save(model.state_dict(), './weights_%s.net' % (session_string))

	# main training loop in the current epoch
	for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in trainset_loader:

		log_probs = model(correspondences.cuda())
		probs = torch.exp(log_probs).cpu()
		target_probs = torch.zeros(probs.size())

		for b in range(0, correspondences.size(0)):

			# calculate the target distribution (see paper supplement A, Eq. 12)
			if opt.fmat:

				# === CASE FUNDAMENTAL MATRIX =========================================

				util.denormalize_pts(correspondences[b, 0:2], im_size1[b])
				util.denormalize_pts(correspondences[b, 2:4], im_size2[b])

				ngransac.gtdist(correspondences[b], target_probs[b], gt_F[b], opt.threshold, True)
			else:

				# === CASE ESSENTIAL MATRIX =========================================

				ngransac.gtdist(correspondences[b], target_probs[b], gt_E[b], opt.threshold, False)

		log_probs.squeeze_()
		target_probs.squeeze_()

		# KL divergence
		loss = distLoss(log_probs, target_probs.cuda()) / correspondences.size(0) 

		# update model
		loss.backward()
		optimizer.step() 
		optimizer.zero_grad()

		print("Iteration: ", iteration, "Loss: ", float(loss))
		train_log.write('%d %f\n' % (iteration, loss))

		iteration += 1
		
		del log_probs, probs, target_probs, loss
