#pragma once
#include <opencv2/core.hpp>
#include "Enums.h"

using namespace std;

/**
* Performs classifier tests.
*/
class Testing {
public:

	/**
	* Compute classification scores for the given class against the other classes.
	  \param classes - real class labels
	  \param responses - class labels from classifier
	  \param cl - label of class, for which scores are computed
	  \return set of scores in the following order: TP, FP, TN, FN, tp rate, fp rate, precision, accuracy, specifity
	*/
	static cv::Mat test(cv::Mat classes, cv::Mat responses, int cl);

	/**
	* Print test scores obtained from "test" method.
	  \param tests - set of scores obtained from "test" method
	*/
	static void printTestResults(cv::Mat tests);

	/**
	* Performs cross validation on the given set.
	  \param set - set of attributes - each row is treated as one sample
	  \param classes - set of class labels corresponding with dataset samples
	  \param method - classification method
	  \param classesCnt - how many classes are present in dataset
	  \param k - into how many subsets the original set should be divided
	*/
	static cv::Mat crossValidation(cv::Mat set, cv::Mat classes, Classification method, int classesCnt, int k = 10);
};