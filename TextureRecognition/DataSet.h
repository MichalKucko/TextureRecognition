#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "FeatureExtractor.h"
#include "Preprocessing.h"
#include "Binarization.h"

using namespace std;

/**
* Class used for preparing dataset containing samples of features and corresponding class labels. 
* By default it is assumed, that images, from which the set is created, are placed in the location of 
* executable or "main.c" file in folder called "database" and then in folders associated with different classes.
* For example, if in "database" folder there are two folders called "A" and "B", images in these folders represent two classes - A and B.
*/
class DataSet {
private:
	
	/**
	* path to folders containing images
	*/
	string learnTestPath = "./database/set_learn";

	string classifyPath = "./database/set_classify";
	
	int imreadFlag = cv::IMREAD_COLOR;

	/**
	* set of features get from images
	*/
	cv::Mat set;
	
	/**
	* set of image class labels
	*/
	cv::Mat classes;

	/**
	* randomly shuffled set of features
	*/
	cv::Mat shuffledSet;

	/**
	* class labels ordered as samples in shuffledSet
	*/
	cv::Mat shuffledClasses;

	/**
	* set of features used for classifier training
	*/
	cv::Mat learnSet;

	/**
	* set of class labels used for classifier training
	*/
	cv::Mat learnClasses;

	/**
	* set of features used for classifier testing
	*/
	cv::Mat testSet;
	
	/**
	* set of class labels used for classifier testing
	*/
	cv::Mat testClasses;

	vector<cv::Mat> segmentImgs;

	vector<cv::Mat> classifySet;

	vector<string> classNames;

	/**
	* vector of samples indices - internal variable used for shuffling dataset
	*/
	vector<int> indices;

	vector<Preprocessing*> preps;
	vector<Binarization*> bins;
	vector<Preprocessing*> binPreps;

public:

	/**
	* Sets path to folders containing images.
	*/
	void setLearnTestPath(string path) { learnTestPath = path; }

	void setClassifyPath(string path) { classifyPath = path; }


	/**
	* Returns the whole dataset.
	*/
	cv::Mat getSet() { return set; }

	/**
	* Returns set used for learning.
	*/
	cv::Mat getLearnSet() { return learnSet; }

	/**
	* Returns set used for testing.
	*/
	cv::Mat getTestSet() { return testSet; }

	/**
	* Returns randomly shuffled set.
	*/
	cv::Mat getShuffledSet() { return shuffledSet; }

	/**
	* Returns all class labels.
	*/
	cv::Mat getClasses() { return classes; }

	/**
	* Returns class labels used for learning.
	*/
	cv::Mat getLearnClasses() { return learnClasses; }

	/**
	* Returns class labels used for testing.
	*/
	cv::Mat getTestClasses() { return testClasses; }

	/**
	* Returns class labels ordered as samples in shuffledSet.
	*/
	cv::Mat getShuffledClasses() { return shuffledClasses; }

	vector<cv::Mat> getClassifySet() { return classifySet;  }

	vector<cv::Mat> getSegmentImgs() { return segmentImgs; }

	void addPreprocess(Preprocessing *prep) {
		preps.push_back(prep);
	}

	void addBinarization(Binarization *bin) {
		bins.push_back(bin);
	}

	void addBinPreprocess(Preprocessing *prep) {
		binPreps.push_back(prep);
	}

	/**
	* Reads images, optionally performs image preprocessing and extracts set of features.
	  \param - preprocessType method used in image preprocessing
	  \param - featuresType type of image features to be extracted
	*/
	int prepareLearnTestSet(FeatureExtractor *feat);
	
	void prepareClassifySet(FeatureExtractor *feat, int minArea = 1000, int segmentsCnt = 10, bool show = false);

	/**
	* Randomly shuffles dataset and place the results in "shuffledSet" field.
	*/
	void shuffleSet();

	/**
	* Splits dataset into learning set and testing set in the given ratio. 
	* Sets are obtainable in "learnSet" and "testSet" fields.
	  \param - learnToTestRatio ratio of lerning set size to testing set size
	*/
	void splitSet(float learnToTestRatio);

	void clearSets();

	void displayClassNames();

	void clearMethods();
};