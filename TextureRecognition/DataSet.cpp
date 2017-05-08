#include "DataSet.h"
#include "Segmentation.h"
#include <iostream>
#include "opencv2_Directory.h"
#include <opencv2/highgui.hpp>


int DataSet::prepareLearnTestSet(FeatureExtractor *feat) {
	vector<string> folders = cv::Directory::GetListFolders(learnTestPath);
	vector<string> files, failedFiles;
	cv::Mat img, features;
	int i = 0, classIndex = 1, lastSlash;

	for (string folder : folders) {
		files = cv::Directory::GetListFiles(folder);
		for (string file : files) {
			
			img = cv::imread(file, imreadFlag);
			if (img.empty()) {
				failedFiles.push_back(file);
			}
			else {

				for (Preprocessing *prep : preps) {
					prep->preprocess(img, img);
				}

				features = feat->extract(img);
				set.push_back(features);
				classes.push_back(classIndex);
				indices.push_back(i);
				i++;
			}
		}
		lastSlash = folder.find_last_of('/');
		classNames.push_back(folder.substr(lastSlash + 1, folder.size() - lastSlash - 1));
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


void DataSet::prepareClassifySet(FeatureExtractor *feat, int minArea, int segmentsCnt, bool show) {
	vector<string> files, failedFiles;
	cv::Mat img, features, binImg, binTmp, segmentsImg, mask;
	Segmentation seg(minArea, segmentsCnt);
	int segments, i = 0;
	files = cv::Directory::GetListFiles(classifyPath);

	for (string file : files) {

		img = cv::imread(file, imreadFlag);
		if (img.empty()) {
			failedFiles.push_back(file);
		}
		else {

			if (show)
				imshow("Image" + to_string(i), img);

			binImg = cv::Mat::zeros(img.size(), CV_8U);
			for (Preprocessing *prep : preps) {
				prep->preprocess(img, img);
			}

			if (!bins.empty()) {
				for (Binarization *bin : bins) {
					bin->prepareImage(img);
					bin->binarize(binTmp);
					cv::bitwise_or(binImg, binTmp, binImg);
				}

				for (Preprocessing *binPrep : binPreps) {
					binPrep->preprocess(binImg, binImg);
				}
				//cv::imshow("Bin" + to_string(i), binImg);
				segments = seg.findSegments(binImg, segmentsImg, show, "Segments" + to_string(i));

				if (segments) {
					//cv::imshow("Segments", segments * 255 / segmentsCnt);
					segmentImgs.push_back(segmentsImg.clone());
					classifySet.push_back(cv::Mat());
					for (int s = 1; s <= segments; s++) {
						cv::inRange(segmentsImg, s, s, mask);
						//cv::imshow("mask"+s, mask);
						features = feat->extract(img, mask);
						classifySet[i].push_back(features);
					}
					i++;
				}
			}
		}
	}

	if (!failedFiles.empty()) {
		cout << "Failed to load following files:" << endl;
		for (string failedFile : failedFiles) {
			cout << failedFile << endl;
		}
		cout << endl;
	}

	cout << "Successfully loaded " << classifySet.size() << " files." << endl << endl;
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
	if (learnToTestRatio == 0) {
		learnSet = set;
		learnClasses = classes;
		return;
	}
	int cut = int(set.rows * learnToTestRatio / (learnToTestRatio + 1));
	learnSet = shuffledSet.rowRange(0, cut);
	testSet = shuffledSet.rowRange(cut, set.rows);
	learnClasses = shuffledClasses.rowRange(0, cut);
	testClasses = shuffledClasses.rowRange(cut, set.rows);
}


void DataSet::displayClassNames() {
	string name;
	cout << "Class names:" << endl;
	for (int i = 0; i < classNames.size(); i++) {
		name = classNames[i];
		cout << i + 1 << " " << name << endl;
	}
}


void DataSet::clearSets() {
	set.release();
	learnSet.release();
	testSet.release();
	shuffledSet.release();
	classifySet.clear();
	classes.release();
	learnClasses.release();
	testClasses.release();
	shuffledClasses.release();
	segmentImgs.clear();
	classNames.clear();
	indices.clear();
}


void DataSet::clearMethods() {
	for (Preprocessing* prep : preps) {
		delete prep;
	}
	for (Binarization* bin : bins) {
		delete bin;
	}
	for (Preprocessing* binPrep : binPreps) {
		delete binPrep;
	}

	preps.clear();
	bins.clear();
	binPreps.clear();
}