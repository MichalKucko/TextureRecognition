#include "FeatureExtractor.h"
#include "DataSet.h"
#include "kNNClassifier.h"
#include "Testing.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	srand(time(NULL));
	DataSet set;
	int classCnt = set.prepareSet(NONE, SIMPLE_DOM_COL);
	set.shuffleSet();

	/*set.splitSet(3);
	kNNClassifier kNN(3, cv::ml::KNearest::BRUTE_FORCE);
	kNN.train(set.getLearnSet(), set.getLearnClasses());
	cv::Mat responses = kNN.classify(set.getTestSet());
	cv::Mat tests = Testing::test(set.getTestClasses(), responses, 1);
	cout << set.getTestClasses() << endl;
	cout << responses << endl;*/

	Testing::crossValidation(set.getShuffledSet(), set.getShuffledClasses(), KNN, classCnt, 3);

	waitKey(0); // Wait for a keystroke in the window
	getchar();
	return 0;
}