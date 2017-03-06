#include "kNNClassifier.h"


void kNNClassifier::train(cv::Mat set, cv::Mat classes) {
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(set, cv::ml::ROW_SAMPLE, classes);
	kNN = cv::ml::KNearest::create();
	kNN->setIsClassifier(true);
	kNN->setAlgorithmType(algorithm);
	kNN->train(set, cv::ml::ROW_SAMPLE, classes);
}


cv::Mat kNNClassifier::classify(cv::Mat set) {
	cv::Mat classes;
	kNN->findNearest(set, k, classes);
	return classes;
}