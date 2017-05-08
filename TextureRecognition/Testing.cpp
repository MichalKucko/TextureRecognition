#include "Testing.h"
#include <iostream>
#include <opencv2/highgui.hpp>


void Testing::test(cv::Mat classes, cv::Mat responses, int cl, cv::Mat &tests, bool push) {
	float tp = 0, tn = 0, fp = 0, fn = 0, p = 0, n = 0;
	int class_i, response_i;
	for (int i = 0; i < classes.rows; i++) {
		class_i = classes.row(i).at<int>(0);
		response_i = (int)responses.row(i).at<float>(0);

		if (class_i == response_i && class_i == cl) {
			tp++;
			p++;
		}
		else if (class_i == response_i && class_i != cl) {
			tn++;
			n++;
		}
		else if (class_i != response_i && class_i == cl) {
			fn++;
			p++;
		}
		else {
			fp++;
			n++;
		}
	}

	float tpRate = 0;
	if(tp + fn) tpRate = tp / (tp + fn);
	float fpRate = 1;
	if(fp + tn) fpRate = fp / (fp + tn);
	float precision = 0;
	if (tp + fp) precision = tp / (tp + fp);
	float accuracy = (tp + tn) / (p + n);
	float specifity = 0;
	if (fp + tn) specifity = tn / (fp + tn);

	cv::Mat testsCol;
	testsCol.push_back(tp);
	testsCol.push_back(fp);
	testsCol.push_back(tn);
	testsCol.push_back(fn);
	testsCol.push_back(tpRate);
	testsCol.push_back(fpRate);
	testsCol.push_back(precision);
	testsCol.push_back(accuracy);
	testsCol.push_back(specifity);
	
	if (push)
		tests.push_back(testsCol.t());
	else
		tests = testsCol.clone().t();
}


void Testing::printTestResults(cv::Mat tests) {
	for (int i = 0; i < tests.rows; i++) {
		cout << "Tests for class " << i+1 << ":" << endl;
		cout << "TP: " << tests.row(i).at<float>(0) << endl;
		cout << "FP: " << tests.row(i).at<float>(1) << endl;
		cout << "TN: " << tests.row(i).at<float>(2) << endl;
		cout << "FN: " << tests.row(i).at<float>(3) << endl;
		cout << "tp rate: " << tests.row(i).at<float>(4) << endl;
		cout << "fp rate: " << tests.row(i).at<float>(5) << endl;
		cout << "precision: " << tests.row(i).at<float>(6) << endl;
		cout << "accuracy: " << tests.row(i).at<float>(7) << endl;
		cout << "specifity: " << tests.row(i).at<float>(8) << endl << endl;
	}
}


cv::Mat Testing::crossValidation(cv::Mat set, cv::Mat classes, Classifier *classifier, int classesCnt, int k) {	
	int step = set.rows / k;
	cv::Mat responses, tests, singleTest;
	for (int i = 0; i < k; i++) {
		cv::Mat learnSet, testSet, learnClasses, testClasses;
		if (i == 0) {
			learnSet = set.rowRange((i + 1) * step, set.rows);
			testSet = set.rowRange(0, (i + 1) * step);
			learnClasses = classes.rowRange((i + 1) * step, set.rows);
			testClasses = classes.rowRange(0, (i + 1) * step);
		}
		else if (i < k - 1) {
			cv::vconcat(set.rowRange(0, i * step), set.rowRange((i + 1) * step, set.rows), learnSet);
			testSet = set.rowRange(i * step, (i + 1) * step);
			cv::vconcat(classes.rowRange(0, i * step), classes.rowRange((i + 1) * step, set.rows), learnClasses);
			testClasses = classes.rowRange(i * step, (i + 1) * step);
		}
		else {
			learnSet = set.rowRange(0, i * step);
			testSet = set.rowRange(i * step, set.rows);
			learnClasses = classes.rowRange(0, i * step);
			testClasses = classes.rowRange(i * step, set.rows);
		}

		classifier->train(learnSet, learnClasses);
		responses = classifier->classify(testSet);
		//cout << classes << endl << endl;
		//cout << testClasses << endl;
		//cout << responses << endl << endl;

		if (i == 0) {
			for (int j = 1; j <= classesCnt; j++) {
				test(testClasses, responses, j, tests);
			}
			//cout << tests << endl;
		}
		else {
			for (int j = 1; j <= classesCnt; j++) {
				test(testClasses, responses, j, singleTest, false);
				tests.row(j-1) += singleTest;
			}
			//cout << tests << endl;
		}
	}

	tests /= k;
	printTestResults(tests);

	return tests;
}


void Testing::displayClassifiedSegments(vector<cv::Mat> &set, vector<cv::Mat> &segmentImgs, Classifier *classifier, int maxClass) {
	for (int i = 0; i < set.size(); i++) {
		classifier->classifySegments(set[i], segmentImgs[i], maxClass);
		cv::imshow("Classified" + to_string(i), segmentImgs[i]);
	}
}