import numpy as np
import argparse
import util

parser = argparse.ArgumentParser(
		description="Calculate the mean AUC over the datasets used in the NG-RANSAC paper, Sec. 4.1",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--fmat', '-fmat', action='store_true', 
	help='estimate the fundamental matrix, instead of the essential matrix')

parser.add_argument('--rootsift', '-rs', action='store_true', 
	help='use RootSIFT normalization')

parser.add_argument('--orb', '-orb', action='store_true', 
	help='use ORB instead of SIFT')

parser.add_argument('--ratio', '-r', type=float, default=1.0, 
		help='apply Lowes ratio filter with the given ratio threshold, 1.0 does nothing')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files, useful to separate different runs of a script')

opt = parser.parse_args()

# evaluation file name
session_string = util.create_session_string('test', opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session)
file = session_string + '.txt'

# folder containing evaluation results
folder = 'results/'

def dataset_mean(datasets):
	'''Calculate the mean evaluation results over a list of datasets.'''
	mean = None

	for ds in datasets:

		# construct complete evaluation file name including base folder and dataset
		result = np.loadtxt(folder + ds + '/' + file)

		if mean is None:
			mean = result
		else:
			mean += result

	return mean / len(datasets)


mean_outdoor = dataset_mean(util.outdoor_test_datasets)
mean_indoor = dataset_mean(util.indoor_test_datasets)
mean = dataset_mean(util.test_datasets)

print("AUC (5deg/10deg/20deg)")
print("Outdoor: %.2f/%.2f/%.2f" % (mean_outdoor[0],mean_outdoor[1],mean_outdoor[2]))
print("Indoor: %.2f/%.2f/%.2f" % (mean_indoor[0],mean_indoor[1],mean_indoor[2]))
print("All: %.2f/%.2f/%.2f" % (mean[0],mean[1],mean[2]))
