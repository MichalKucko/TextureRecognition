#include "kNNClassifier.h"
#include <opencv2/highgui.hpp>


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


cv::Mat kNNClassifier::classifySegments(cv::Mat set, cv::Mat &segmentsImg, int maxClass) {
	cv::Mat classes;
	kNN->findNearest(set, k, classes);
	uchar pixel;
	//cv::imshow("Segments", segmentsImg * 255 / maxClass);
	//cv::waitKey(0);
	//cout << set << endl;
	//cout << classes << endl;
	for (int x = 0; x < segmentsImg.rows; x++) {
		for (int y = 0; y < segmentsImg.cols; y++) {
			pixel = segmentsImg.at<uchar>(x, y);
			if (pixel) 
				segmentsImg.at<uchar>(x, y) = classes.row(pixel - 1).at<float>(0) * 255 / maxClass;
		}
	}
	return classes;
}