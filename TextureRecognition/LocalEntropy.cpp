#include "LocalEntropy.h"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

using namespace std;

void LocalEntropy::prepareImage(cv::Mat &src, cv::Mat &dst) {
	img = cv::Mat(src.size(), CV_8U);
	cv::Mat grayImg;
	if(src.type() != CV_8U && src.type() != CV_8UC1)
		cv::cvtColor(src, grayImg, cv::COLOR_BGR2GRAY);
	
	float range[] = { 0, 256 };
	const float* histRange = { range };
	float sum = size * size;
	float bin, prob, entropy = 0;
	cv::Rect roi;
	cv::Mat hist;
	int offset = size / 2;
	float maxEntropy = log2(histSize);
	int widthX, widthY;
	
	for (int x = offset; x < grayImg.cols - offset; x++) {
		for (int y = offset; y < grayImg.rows - offset; y++) {
			entropy = 0;
			//widthX = min(min(size, size + x - offset), src.cols - x);
			//widthY = min(min(size, size + y - offset), src.rows - y);
			//widthX = min(size, size + x - offset);
			//widthY = min(size, size + y - offset);
			//roi = cv::Rect(max(0, min(x - offset, x - size)), max(0, min(y - offset, y - size)), widthX, widthY);
			roi = cv::Rect(x - offset, y - offset, size, size);
			cv::Mat hist;
			cv::calcHist(&grayImg(roi), 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
			for (int i = 0; i < hist.rows; i++) {
				bin = hist.at<float>(i);
				if (bin) {
					prob = bin / sum;
					entropy -= prob * log2(prob);
				}
			}
			img.at<uchar>(y, x) = entropy * 255 / maxEntropy;
		}
	}

	for (int i = 0; i < offset; i++) {
		img.row(offset).copyTo(img.row(i));
	}
	for (int i = grayImg.rows - offset; i < grayImg.rows; i++) {
		img.row(grayImg.rows - offset - 1).copyTo(img.row(i));
	}
	for (int i = 0; i < offset; i++) {
		img.col(offset).copyTo(img.col(i));
	}
	for (int i = grayImg.cols - offset; i < grayImg.cols; i++) {
		img.col(grayImg.cols - offset - 1).copyTo(img.col(i));
	}

	dst = img;
	cv::imshow("Entropy", img);
	if (adaptiveThres)
		thres = meanMultiplier * cv::mean(img)[0];
}