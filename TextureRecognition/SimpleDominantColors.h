#pragma once
#include "FeatureExtractor.h"

/**
* Extracts largest values from color histogram.
*/
class SimpleDominantColors : public FeatureExtractor {
private:

	/**
	* number of histogram bins
	*/
	int histSize;

	/**
	* number of values returned by extractor
	*/
	int howManyColors;

public:

	SimpleDominantColors() : histSize(100), howManyColors(3) {};

	/**
	\param _histSize - number of histogram bins
	\param _howManyColors - number of values returned by extractor
	*/
	SimpleDominantColors(int _histSize, int _howManyColors) : histSize(_histSize), howManyColors(_howManyColors) {};

	/**
	\param _histSize - number of histogram bins
	\param _howManyColors - number of values returned by extractor
	*/
	void setParams(int _histSize, int _howManyColors);

	cv::Mat extract(cv::Mat img);
};