#include "Testing.h"
#include "kNNClassifier.h"


cv::Mat Testing::test(cv::Mat classes, cv::Mat responses, int cl) {
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

	cv::Mat tests;
	tests.push_back(tp);
	tests.push_back(fp);
	tests.push_back(tn);
	tests.push_back(fn);
	tests.push_back(tpRate);
	tests.push_back(fpRate);
	tests.push_back(precision);
	tests.push_back(accuracy);
	tests.push_back(specifity);

	return tests.t();
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


cv::Mat Testing::crossValidation(cv::Mat set, cv::Mat classes, Classification method, int classesCnt, int k) {	
	Classifier *classifier = new Classifier();
	if (method == KNN)
		classifier = new kNNClassifier();

	int step = set.rows / k;
	cv::Mat responses, tests;
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
				tests.push_back(test(testClasses, responses, j));
			}
			//cout << tests << endl;
		}
		else {
			for (int j = 1; j <= classesCnt; j++) {
				tests.row(j-1) += test(testClasses, responses, j);
			}
			//cout << tests << endl;
		}
	}

	tests /= k;
	printTestResults(tests);

	return tests;
}