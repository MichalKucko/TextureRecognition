#include "CannyBinarization.h"

void CannyBinarization::prepareImage(cv::Mat &src, cv::Mat &dst) {
	if (blurSize) {
		if (gaussSigma)
			cv::GaussianBlur(src, img, cv::Size(blurSize, blurSize), gaussSigma);
		else
			cv::blur(src, img, cv::Size(blurSize, blurSize));
	}
	cv::Canny(src, img, thres1, thres2, apertureSize);
}