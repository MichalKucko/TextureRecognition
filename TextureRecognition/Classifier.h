#pragma once
#include <opencv2/core.hpp>

using namespace std;

/**
* Abstract class, from which inherit specific classifiers classes.
*/
class Classifier {
public:

	/**
	* Trains classifier.
	  \param set - set of attributes - each row is treated as one sample
	  \param classes - set of class labels corresponding with dataset samples
	*/
	virtual void train(cv::Mat set, cv::Mat classes) = 0;

	/**
	* Returns set of class labels predicted by classifier for given dataset.
	\param set - set of attributes - each row is treated as one sample
	*/
	virtual cv::Mat classify(cv::Mat set) = 0;

	virtual cv::Mat classifySegments(cv::Mat set, cv::Mat &segmentsImg, int maxClass) = 0;
};