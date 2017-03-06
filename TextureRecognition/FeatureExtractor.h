#pragma once
#include <opencv2/core.hpp>

using namespace std;

/**
* Abstract class, from which inherit specific feature extractor classes.
*/
class FeatureExtractor {
public:

	/**
	* Returns set of features extracted from given image.
	  \param img - image
	*/
	virtual cv::Mat extract(cv::Mat img) { return cv::Mat(); }
};