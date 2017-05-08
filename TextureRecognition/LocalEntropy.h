#pragma once
#include "Binarization.h"


/**
*
*/
class LocalEntropy : public Binarization {
private:

	/**
	* filter size
	*/
	int size;

	/**
	* number of histogram bins
	*/
	int histSize;

	float meanMultiplier;

public:

	LocalEntropy() : size(9), histSize(50), meanMultiplier(1) {};

	/**
	\param size - filter size
	*/
	LocalEntropy(int _size, int _histSize, float _meanMultiplier = 1) : size(_size), histSize(_histSize), meanMultiplier(_meanMultiplier) {};

	/**
	* Sets filter size.
	\param size - filter size
	*/
	void setSize(int _size) { size = _size; };

	/**
	* Sets number of histogram bins.
	\param size - number of histogram bins
	*/
	void setHistSize(int _histSize) { histSize = _histSize; };

	void setMeanMultiplier(float _meanMultiplier) { meanMultiplier = _meanMultiplier; }

	void prepareImage(cv::Mat &src, cv::Mat &dst = cv::Mat());
};