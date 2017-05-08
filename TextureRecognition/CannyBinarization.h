#pragma once
#include "Binarization.h"

class CannyBinarization : public Binarization {
private:

	int blurSize;
	float gaussSigma;
	int thres1, thres2;
	int apertureSize;
	bool L2Gradient;

public:

	CannyBinarization() : blurSize(3), gaussSigma(0), thres1(50), thres2(150), apertureSize(3), L2Gradient(false) {};
	
	CannyBinarization(int bS, float gS, int t1, int t2, int aS = 3, bool L2 = false) : 
		blurSize(bS), gaussSigma(gS), thres1(t1), thres2(t2), apertureSize(aS), L2Gradient(L2) {};
	
	CannyBinarization(int bS, float gS, int t1, float ratio = 3, int aS = 3, bool L2 = false) :
		blurSize(bS), gaussSigma(gS), thres1(t1), thres2(t1 * ratio), apertureSize(aS), L2Gradient(L2) {};

	void binarize(cv::Mat &dst) { dst = img; }

	void prepareImage(cv::Mat &src, cv::Mat &dst = cv::Mat());
};