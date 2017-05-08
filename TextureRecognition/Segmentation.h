#pragma once
#include <opencv2/core.hpp>

using namespace std;

class Segmentation {
private:
	int minArea, segmentsCnt;
	
	static bool largerArea(vector<cv::Point> &i, vector<cv::Point> &j) { return (i.size() > j.size()); }

public:
	Segmentation() : minArea(1000), segmentsCnt(10) {};
	Segmentation(int _minArea, int _segmentsCnt) : minArea(_minArea), segmentsCnt(_segmentsCnt) {};

	int findSegments(cv::Mat &binaryImg, cv::Mat &segmentsImg, bool display = false, string title = "Segments");
	//void displaySegments(cv::Mat &img, vector<vector<cv::Point>> &contours, bool save = false);
};