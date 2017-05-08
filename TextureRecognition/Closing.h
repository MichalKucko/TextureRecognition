#pragma once
#include "Preprocessing.h"
#include <opencv2/imgproc/imgproc.hpp>

class Closing : public Preprocessing {
private:

	int size;

	int iterations;

public:

	Closing() : size(3), iterations(1) {};
	Closing(int _size, int _iterations) : size(_size), iterations(_iterations) {};

	void preprocess(cv::Mat &src, cv::Mat &dst) {
		cv::Mat mask = cv::Mat::ones(size, size, CV_8U);
		cv::morphologyEx(src, dst, cv::MORPH_CLOSE, mask, cv::Point(-1,-1), iterations);
	}
};