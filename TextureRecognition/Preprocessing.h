#pragma once
#include <opencv2/core.hpp>

/**
* Abstract class, from which inherit specific image preprocessing classes.
*/
class Preprocessing {
public:

	/**
	* Applies preprocessing.
	  \param img - image
	*/
	virtual void preprocess(cv::Mat &src, cv::Mat &dst) = 0;
};