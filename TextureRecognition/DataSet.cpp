#include "DataSet.h"
#include "opencv2_Directory.h"
#include "FeatureExtractor.h"
#include "SimpleBlur.h"
#include "SimpleDominantColors.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


int DataSet::prepareSet(Preprocess preprocessType, Feature featuresType) {
	
	Preprocessing *prep = new Preprocessing();
	if (preprocessType == SIMPLE_BLUR)
		prep = new SimpleBlur();

	FeatureExtractor *featEx = new FeatureExtractor();
	if (featuresType == SIMPLE_DOM_COL)
		featEx = new SimpleDominantColors();

	vector<string> folders = cv::Directory::GetListFolders(imagesPath);
	vector<string> files, failedFiles;
	cv::Mat img, features;
	int i = 0, classIndex = 1;
	for (string folder : folders) {
		files = cv::Directory::GetListFiles(folder);
		for (string file : files) {
			
			img = cv::imread(file, cv::IMREAD_COLOR);
			if (img.empty()) {
				failedFiles.push_back(file);
			}
			else {
				if (preprocessType != NONE)
					prep->preprocess(img);
				features = featEx->extract(img);
				set.push_back(features);
				classes.push_back(classIndex);
				indices.push_back(i);
				i++;
			}
		}
		classIndex++;
	}

	if (!failedFiles.empty()) {
		cout << "Failed to load following files:" << endl;
		for (string failedFile : failedFiles) {
			cout << failedFile << endl;
		}
		cout << endl;
	}

	cout << "Successfully loaded " << set.rows << " files." << endl << endl;

	return classIndex - 1;
}


void DataSet::shuffleSet() {
	//cv::randShuffle(indices);
	random_shuffle(indices.begin(), indices.end());
	for (int i : indices) {
		shuffledSet.push_back(set.row(i));
		shuffledClasses.push_back(classes.row(i));
	}
}


void DataSet::splitSet(float learnToTestRatio) {
	int cut = int(set.rows * learnToTestRatio / (learnToTestRatio + 1));
	learnSet = shuffledSet.rowRange(0, cut);
	testSet = shuffledSet.rowRange(cut, set.rows);
	learnClasses = shuffledClasses.rowRange(0, cut);
	testClasses = shuffledClasses.rowRange(cut, set.rows);
}