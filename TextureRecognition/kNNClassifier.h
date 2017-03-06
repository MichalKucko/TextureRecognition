#pragma once
#include "Classifier.h"
#include <opencv2/ml.hpp>

/**
* k Nearest Neighbours classifier
*/
class kNNClassifier : public Classifier {
private:

	/**
	* number of naighbours used in classification
	*/
	int k;

	/**
	* classification algorithm
	*/
	cv::ml::KNearest::Types algorithm;

	/**
	* classifier object - internal variable
	*/
	cv::Ptr<cv::ml::KNearest> kNN;

public:

	kNNClassifier() : k(3), algorithm(cv::ml::KNearest::BRUTE_FORCE) {};

	/**
	  \param _k - number of naighbours used in classification
	  \param _alg - classification algorithm
	*/
	kNNClassifier(int _k, cv::ml::KNearest::Types _alg) : k(_k), algorithm(_alg) {};
	
	/**
	\param _k - number of naighbours used in classification
	*/
	void setK(int _k) { k = _k;  }

	void train(cv::Mat set, cv::Mat classes);

	cv::Mat classify(cv::Mat set);
};