#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <omp.h>

#include <iostream>

#include "thread_rand.h"
#include "opencv-five-point.h"
#include "opencv-fundam.h"


/**
* @brief Find essential matrix given 2D-2D correspondences.
* 
* Method uses RANSAC with guided sampling (NG-RANSAC) and the 5-point algorithm of OpenCV.
* 
* @param correspondences 3-dim float tensor, dim 1: xy coordinates of left and right image (normalized by calibration parameters), dim 2: correspondences, dim 3: dummy dimension
* @param probabilities 3-dim float tensor, dim 1: sampling weight, dim 2: correspondences, dim 3: dummy dimension
* @param randSeed random seed
* @param hypCount number of hypotheses/iterations for RANSAC
* @param inlierThresh inlier threshold (in normalized image coordinates)
* @param out_E 2-dim float tensor, estimated essential matrix
* @param out_inliers, 3-dim float tensor, marks the inliers of the final model, dim 1: inlier or not, dim 2: correspondences, dim 3: dummy dimension
* @param out_gradients, 3-dim float tensor, marks the minimal sets sampled for calculation of the neural guidance gradients, dim 1: has been sampled or not, dim 2: correspondences, dim 3: dummy dimension
* @return double inlier count of best model
*/
double find_essential_mat(
	at::Tensor correspondences,
	at::Tensor probabilities,
	int randSeed,
	int hypCount,
	float inlierThresh,
	at::Tensor out_E,
	at::Tensor out_inliers,
	at::Tensor out_gradients
	)
{
	float computeErrT = inlierThresh * inlierThresh; // we will calculate the squared error

	int cCount = probabilities.size(1); // number of correspondences
	int cMin = 5; // size of the minimal set

	// access PyTorch tensors
	at::TensorAccessor<float, 3> cAccess = correspondences.accessor<float, 3>();		
	at::TensorAccessor<float, 3> pAccess = probabilities.accessor<float, 3>();		
	
	at::TensorAccessor<float, 2> EAccess = out_E.accessor<float, 2>();		

	at::TensorAccessor<float, 3> gAccess = out_gradients.accessor<float, 3>();
	at::TensorAccessor<float, 3> iAccess = out_inliers.accessor<float, 3>();	

	// read correspondences and weights
	std::vector<cv::Point2d> pts1, pts2;
	std::vector<float> wPts;

	for(int c = 0; c < cCount; c++)	
	{
		pts1.push_back(cv::Point2d(cAccess[0][c][0], cAccess[1][c][0]));
		pts2.push_back(cv::Point2d(cAccess[2][c][0], cAccess[3][c][0]));
		wPts.push_back(pAccess[0][c][0]);
	}

	// create categorial distribution from weights
	ThreadRand::init(randSeed);
	std::discrete_distribution<int> multinomialDist(wPts.begin(), wPts.end());
	
	cv::Mat_<double> K = cv::Mat_<double>::eye(3, 3); // dummy calibration matrix (assuming normalized coordinates)
	
	std::vector<cv::Mat_<double>> hyps(hypCount); // model hypotheses
	std::vector<cv::Mat> masks(hypCount); // inlier masks
	std::vector<int> inlierCounts(hypCount, -1); // hypothesis scores
	std::vector<std::vector<int>> minSets(hypCount); // minimal sets corresponding to each hypothesis

	// main RANSAC loop
	#pragma omp parallel for
	for(int h = 0; h < hypCount; h++)
	{
		unsigned threadID = omp_get_thread_num();

		//sample a minimal set
		std::vector<cv::Point2d> minSet1(cMin); // coordinates of image 1
		std::vector<cv::Point2d> minSet2(cMin); // coordinates of image 2
		minSets[h] = std::vector<int>(cMin); // mark which correspondences were selected

		for(int j = 0; j < cMin; j++)
		{
			// choose a correspondence based on the provided weights/probabilities
			int cIdx = multinomialDist(ThreadRand::generators[threadID]);

			minSet1[j] = pts1[cIdx];
			minSet2[j] = pts2[cIdx];
			minSets[h][j] = cIdx;
		}

		// call OPENCV find E-mat on the minimal set (using dummy RANSAC parameters)
		cv::Mat mask;
		cv::Mat E = cv::findEssentialMat(minSet1, minSet2, K, cv::RANSAC, 0.01, 1.0, mask);

		// possibly multiple solutions
		int countE = E.rows / 3;

		if(countE == 0)
		{
			// no solution
			E = cv::Mat_<double>::eye(3, 3);
			countE = 1;
		}

		// select best solution according to inlier count
		for(int e = 0; e < countE; e++)
		{
			cv::Mat curE = E.rowRange(e*3, e*3+3);
			// compute distance of each correspondence to the estimated model
			cv::Mat err;
			compute_essential_error(pts1, pts2, curE, err);

			// apply threshold on distance for inliers
			cv::Mat curMask = err < computeErrT;
			int curInlierCount = cv::countNonZero(curMask);

			// store best of the multiple solution 
			if(curInlierCount > inlierCounts[h])
			{
				inlierCounts[h] = curInlierCount;
				masks[h] = curMask;
				hyps[h] = curE;
			}
		}

	}

	int bestScore = -1; // best inlier count
	cv::Mat_<double> bestE; // best model
	cv::Mat bestM; // inlier mask of model

	for(int h = 0; h < hypCount; h++)
	{
		// store best solution overall
		if(inlierCounts[h] > bestScore)
		{
			bestScore = inlierCounts[h];
			bestE = hyps[h];
			bestM = masks[h];
		}

		// keep track of the minimal sets sampled in the gradient tensor
		for(unsigned c = 0; c < minSets[h].size(); c++)
		{
			int cIdx = minSets[h][c];
			gAccess[0][cIdx][0] += 1;
		}
	}

	// create a mask of inliers correspondences for the best model
	for(int cIdx = 0; cIdx < bestM.rows; cIdx++)
	{
		if(bestM.at<uchar>(cIdx, 0) > 0)
			iAccess[0][cIdx][0] = 1;					
	}	

	// write model PyTorch tensor
	for(unsigned y = 0; y < 3; y++)
	for(unsigned x = 0; x < 3; x++)
		EAccess[y][x] = bestE(y, x);		

	return bestScore;
}


/**
* @brief Calculate a ground truth probability distribution for a set of correspondences.
* 
* Method uses using the distance to a ground truth model which 
* can be an essential matrix or a fundamental matrix.
* For more information, see paper supplement A, Eq. 12
* 
* @param correspondences 3-dim float tensor, dim 1: xy coordinates of left and right image (normalized by calibration parameters when used for an essential matrix, i.e. f_mat=false), dim 2: correspondences, dim 3: dummy dimension
* @param out_probabilities 3-dim float tensor, dim 1: ground truth probability, dim 2: correspondences, dim 3: dummy dimension
* @param gt_model 2-dim float tensor, ground truth model, essential matrix or fundamental matrix
* @param threshold determines the softness of the distribution, the inlier threshold used at test time is a good choice
* @param f_mat indicator whether ground truth model is an essential or fundamental matrix)
* @return void
*/
void gtdist(
	at::Tensor correspondences,
	at::Tensor out_probabilities,
	at::Tensor gt_model,
	float threshold,
	bool f_mat
)
{
	// we compute the sequared error, so we use the squared threshold
	threshold *= threshold;

	int cCount = out_probabilities.size(1); // number of correspondences

	// access to PyTorch tensors
	at::TensorAccessor<float, 3> cAccess = correspondences.accessor<float, 3>();		
	at::TensorAccessor<float, 3> pAccess = out_probabilities.accessor<float, 3>();		
	at::TensorAccessor<float, 2> MAccess = gt_model.accessor<float, 2>();		

	// read correspondences
	std::vector<cv::Point2d> pts1, pts2; // coordinates in image 1 and 2

	for(int c = 0; c < cCount; c++)	
	{
		pts1.push_back(cv::Point2d(cAccess[0][c][0], cAccess[1][c][0]));
		pts2.push_back(cv::Point2d(cAccess[2][c][0], cAccess[3][c][0]));
	}

	// read essential matrix
	cv::Mat_<double> gtModel = cv::Mat_<double>::zeros(3, 3);

	for(int x = 0; x < gtModel.cols; x++)
	for(int y = 0; y < gtModel.rows; y++)
		gtModel(y, x) = MAccess[y][x];

	// compute epipolar errors
	cv::Mat gtErr;
	if(f_mat)
	{
		cv::Mat m1(pts1);
		m1.convertTo(m1, CV_32F);
		cv::Mat m2(pts2);
		m2.convertTo(m2, CV_32F);

		compute_fundamental_error(m1, m2, gtModel, gtErr);
	}
	else
		compute_essential_error(pts1, pts2, gtModel, gtErr);

	// compute ground truth correspondence weights (see paper supplement A, Eq. 12)
	std::vector<float> weights(gtErr.rows);
	float normalizer = std::sqrt(2 * 3.1415926 * threshold);
	float probSum = 0;

	for(int j = 0; j < gtErr.rows; j++)
	{
		weights[j] = std::exp(-gtErr.at<float>(j, 0) / 2 / threshold) / normalizer;
		probSum += weights[j];
	}

	// write out results
	for(int j = 0; j < gtErr.rows; j++)
		pAccess[0][j][0] = weights[j] / probSum;	
}

/**
* @brief Find fundamental matrix given 2D-2D correspondences.
* 
* Method uses RANSAC with guided sampling (NG-RANSAC) and the 7-point algorithm of OpenCV.
* 
* @param correspondences 3-dim float tensor, dim 1: xy coordinates of left and right image (in pixels), dim 2: correspondences, dim 3: dummy dimension
* @param probabilities 3-dim float tensor, dim 1: sampling weight, dim 2: correspondences, dim 3: dummy dimension
* @param randSeed random seed
* @param hypCount number of hypotheses/iterations for RANSAC
* @param inlierThresh inlier threshold (in pixels)
* @param refine if true, model will be refined by running the 8-point algorithm on all inliers
* @param out_F 2-dim float tensor, estimated fundamental matrix
* @param out_inliers, 3-dim float tensor, marks the inliers of the final model, dim 1: inlier or not, dim 2: correspondences, dim 3: dummy dimension
* @param out_gradients, 3-dim float tensor, marks the minimal sets sampled for calculation of the neural guidance gradients, dim 1: has been sampled or not, dim 2: correspondences, dim 3: dummy dimension
* @return double inlier count of best model
*/
double find_fundamental_mat(
	at::Tensor correspondences, 
	at::Tensor probabilities,
	int randSeed,
	int hypCount,
	float inlierThresh,
	bool refine,
	at::Tensor out_F,
	at::Tensor out_inliers,
	at::Tensor out_gradients)
{
	float computeErrT = inlierThresh * inlierThresh; // we will calculate the squared error

	int cCount = probabilities.size(1); // number of correspondences
	int cMin = 7; // size of the minimal set

	// access PyTorch tensors
	at::TensorAccessor<float, 3> cAccess = correspondences.accessor<float, 3>();		
	at::TensorAccessor<float, 3> pAccess = probabilities.accessor<float, 3>();	

	at::TensorAccessor<float, 2> FAccess = out_F.accessor<float, 2>();		

	at::TensorAccessor<float, 3> gAccess = out_gradients.accessor<float, 3>();
	at::TensorAccessor<float, 3> iAccess = out_inliers.accessor<float, 3>();	

	// read correspondences and weights
	std::vector<cv::Point2d> pts1, pts2;
	std::vector<float> wPts;

	for(int c = 0; c < cCount; c++)	
	{
		pts1.push_back(cv::Point2d(cAccess[0][c][0], cAccess[1][c][0]));
		pts2.push_back(cv::Point2d(cAccess[2][c][0], cAccess[3][c][0]));
		wPts.push_back(pAccess[0][c][0]);
	}

	cv::Mat m1(pts1);
	m1.convertTo(m1, CV_32F);
	cv::Mat m2(pts2);
	m2.convertTo(m2, CV_32F);

	// create categorial distribution from weights
	ThreadRand::init(randSeed);
	std::discrete_distribution<int> multinomialDist(wPts.begin(), wPts.end());
	
	std::vector<cv::Mat_<double>> hyps(hypCount); // model hypotheses
	std::vector<cv::Mat> masks(hypCount); // inlier masks
	std::vector<int> inlierCounts(hypCount, -1); // hypothesis scores
	std::vector<std::vector<int>> minSets(hypCount); // minimal sets corresponding to each hypothesis

	// main RANSAC loop
	#pragma omp parallel for
	for(int h = 0; h < hypCount; h++)
	{
		unsigned threadID = omp_get_thread_num();

		//sample a minimal set
		std::vector<cv::Point2d> minSet1(cMin);
		std::vector<cv::Point2d> minSet2(cMin);
		minSets[h] = std::vector<int>(cMin);

		for(int j = 0; j < cMin; j++)
		{
			// 2D location in the subsampled image
			int cIdx = multinomialDist(ThreadRand::generators[threadID]);

			minSet1[j] = pts1[cIdx];
			minSet2[j] = pts2[cIdx];
			minSets[h][j] = cIdx;
		}

		cv::Mat F;
		int countF; // possibly multiple solutions
		try
		{
			// call OPENCV find F-mat on the minimal set
			F = cv::findFundamentalMat(minSet1, minSet2, cv::FM_7POINT);
			// potentially more than one model, select according to inlier count
			countF = F.rows / 3;
		}
		catch(...)
		{
			// error, i.e. no valid solution
			countF = 0;
		}

		if(countF == 0)
		{
			// no solution
			F = cv::Mat_<double>::eye(3, 3);
			countF = 1;
		}

		// select best solution according to inlier count
		for(int e = 0; e < countF; e++)
		{
			cv::Mat curF = F.rowRange(e*3, e*3+3);
			// compute distance of each correspondence to the estimated model
			cv::Mat err;
			compute_fundamental_error(m1, m2, curF, err);

			// apply threshold on distance for inliers
			cv::Mat curMask = err < computeErrT;
			int curInlierCount = cv::countNonZero(curMask);

			// store best of the multiple solution
			if(curInlierCount > inlierCounts[h])
			{
				inlierCounts[h] = curInlierCount;
				masks[h] = curMask;
				hyps[h] = curF;
			}
		}
	}

	int bestScore = -1; // best inlier count
	cv::Mat_<double> bestF; // best model
	cv::Mat bestM; // inlier mask of model

	for(int h = 0; h < hypCount; h++)
	{
		// store best solution overall
		if(inlierCounts[h] > bestScore)
		{
			bestScore = inlierCounts[h];
			bestF = hyps[h];
			bestM = masks[h];
		}

		// keep track of the minimal sets sampled in the gradient tensor
		for(unsigned c = 0; c < minSets[h].size(); c++)
		{
			int cIdx = minSets[h][c];
			gAccess[0][cIdx][0] += 1;
		}

	}

	// create a mask of inliers correspondences for the best model
	for(int cIdx = 0; cIdx < bestM.rows; cIdx++)
	{
		if(bestM.at<uchar>(cIdx, 0) > 0)
			iAccess[0][cIdx][0] = 1;					
	}	

 	if(refine && bestScore > 7)
 	{
 		// recalculate F matrix on all inliers
		std::vector<cv::Point2d> in_pts1, in_pts2;

		for(int c = 0; c < cCount; c++)	
		{
			if(!bestM.at<uchar>(0,c)) continue;
			in_pts1.push_back(cv::Point2d(cAccess[0][c][0], cAccess[1][c][0]));
			in_pts2.push_back(cv::Point2d(cAccess[2][c][0], cAccess[3][c][0]));
		}
		
		bestF = cv::findFundamentalMat(in_pts1, in_pts2, cv::FM_8POINT);
		
		if(bestF.rows == 0)
			bestF = cv::Mat_<double>::eye(3, 3); //refinement failed
	}

	if(bestScore < 7)
		bestF = cv::Mat_<double>::eye(3, 3);

	// write model PyTorch tensor
	for(unsigned y = 0; y < 3; y++)
	for(unsigned x = 0; x < 3; x++)
		FAccess[y][x] = bestF(y, x);

	if(bestScore < 7)
		return 0;

	return bestScore;
}


// register C++ functions for use in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("gtdist", &gtdist, "Ground truth distribution for intialization of NG-RANSAC for essential matrix estimation.");
	m.def("find_essential_mat", &find_essential_mat, "Computes essential matrix from given, normalized correspondences.");
	m.def("find_fundamental_mat", &find_fundamental_mat, "Computes fundamental matrix from given correspondences.");
}
