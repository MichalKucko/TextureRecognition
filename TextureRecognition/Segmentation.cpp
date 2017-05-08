#include "Segmentation.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>


int Segmentation::findSegments(cv::Mat &binaryImg, cv::Mat &segmentsImg, bool display, string title) {
	vector<vector<cv::Point>> contours, contoursNeg;
	vector<cv::Vec4i> hierarchy;
	cv::findContours(binaryImg, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	bitwise_not(binaryImg, binaryImg);
	findContours(binaryImg, contoursNeg, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	contours.insert(contours.end(), contoursNeg.begin(), contoursNeg.end());
	bitwise_not(binaryImg, binaryImg);

	double area;
	for (int i = 0; i < contours.size(); i++) {
		area = contourArea(contours[i]);
		if (area < minArea)
			contours.erase(contours.begin() + i);
	}

	sort(contours.begin(), contours.end(), Segmentation::largerArea);
	int segments = min((int)contours.size(), segmentsCnt);
	if (!segments) return 0;

	cv::Mat markers = cv::Mat::zeros(binaryImg.size(), CV_32SC1);
	for (int i = 0; i < segments; i++) {
		drawContours(markers, contours, i, cv::Scalar::all(i + 1), -1);
	}
	//cv::imshow("before", markers * 65000 / segments);

	cv::watershed(cv::Mat(binaryImg.size(), CV_8UC3), markers);
	markers.convertTo(segmentsImg, CV_8UC1);

	if(display) 
		cv::imshow(title, segmentsImg * 255 / segments);

	return segments;
}

