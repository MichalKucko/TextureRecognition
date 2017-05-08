#include "SimpleDominantColors.h"
#include "opencv2/imgproc.hpp"


void SimpleDominantColors::setParams(int _histSize, int _howManyColors) {
	histSize = _histSize;
	howManyColors = _howManyColors;
}


cv::Mat SimpleDominantColors::extract(cv::Mat img, cv::Mat mask) {
	cv::Mat hsvImg;
	cv::cvtColor(img, hsvImg, CV_BGR2HSV);
	vector<cv::Mat> hsvImgs(3);
	split(img, hsvImgs);

	float range[] = { 0, 180 };
	const float* histRange = { range };
	cv::Mat hueHist;
	cv::calcHist(&hsvImgs[0], 1, 0, mask, hueHist, 1, &histSize, &histRange);

	//cv::Mat dominantCols(1, howManyColors, CV_32F);
	cv::Mat dominantCols;
	cv::Mat indices;
	cv::sortIdx(hueHist, indices, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
	for (int i = 0; i < howManyColors; i++) {
		dominantCols.push_back((float)indices.at<int>(i));
	}

	return dominantCols.t();
}