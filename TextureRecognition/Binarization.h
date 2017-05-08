#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
* Abstract class, from which inherit specific image preprocessing classes.
*/
class Binarization {
protected:
	uchar thres = 0;
	cv::Mat img;
	bool adaptiveThres = false;

public:

	void setThres(int _thres) { thres = _thres; }

	void setAdaptiveThres(bool adaptive) { adaptiveThres = adaptive; }

	void binarize(cv::Mat &dst) {
		cv::threshold(img, dst, thres, 255, CV_THRESH_BINARY);
	}

	virtual void prepareImage(cv::Mat &src, cv::Mat &dst = cv::Mat()) = 0;
};