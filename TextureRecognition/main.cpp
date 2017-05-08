#include "SimpleDominantColors.h"
#include "DataSet.h"
#include "kNNClassifier.h"
#include "Testing.h"
#include "LocalEntropy.h"
#include "Segmentation.h"
#include "CannyBinarization.h"
#include "Closing.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ctime>
#include <forward_list>
#include <iomanip>


using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
	
	srand(time(NULL));
	DataSet set;
	FeatureExtractor *feat = new SimpleDominantColors(100, 6);
	kNNClassifier kNN(3, cv::ml::KNearest::BRUTE_FORCE);
	int classCnt;

	classCnt = set.prepareLearnTestSet(feat);
	set.shuffleSet();
	set.splitSet(3);
	kNN.train(set.getLearnSet(), set.getLearnClasses());
	cv::Mat responses = kNN.classify(set.getTestSet());
	/*cout << set.getTestClasses() << endl;
	cout << responses << endl;
	cout << endl << set.getTestSet() << endl << endl;
	cout << endl;*/
	cv::Mat tests;
	Testing::test(set.getTestClasses(), responses, 1, tests);
	Testing::test(set.getTestClasses(), responses, 2, tests);
	Testing::test(set.getTestClasses(), responses, 3, tests);
	Testing::test(set.getTestClasses(), responses, 4, tests);
	Testing::printTestResults(tests);
	cout << endl << endl;

	cout << "Cross validation:" << endl << endl;
	Testing::crossValidation(set.getShuffledSet(), set.getShuffledClasses(), &kNN, classCnt, 3);

	set.clearSets();

	classCnt = set.prepareLearnTestSet(feat);
	set.splitSet(0);
	kNN.train(set.getLearnSet(), set.getLearnClasses());
	Binarization *entropy = new LocalEntropy(11, 10, 1.1);
	entropy->setAdaptiveThres(true);
	set.addBinarization(entropy);
	Preprocessing *closing = new Closing();
	set.addBinPreprocess(closing);
	set.prepareClassifySet(feat, 5000, 6, true);
	Testing::displayClassifiedSegments(set.getClassifySet(), set.getSegmentImgs(), &kNN, classCnt);
	set.displayClassNames();

	set.clearSets();
	set.clearMethods();


	/*cv::Mat img = cv::imread("./database/field/EP-00-00012_0119_0009 (Custom).JPG", cv::IMREAD_GRAYSCALE);
	//cv::imshow("img", img);
	cv::Mat entropy(img.size(), CV_8U);
	LocalEntropy loc(9, 10, 1.1);
	loc.setAdaptiveThres(true);
	loc.prepareImage(img, entropy);
	cv::imshow("entropy", entropy);
	cv::Mat bin;
	loc.binarize(bin);
	cv::imshow("binary", bin);
	Segmentation seg;
	cv::Mat segments, binCol;
	cvtColor(entropy, entropy, CV_GRAY2BGR);
	cvtColor(bin, binCol, CV_GRAY2BGR);
	seg.findSegments(bin, segments, true, "Segments");*/
	

	/*cv::Mat img = cv::imread("./database/set_classify/img.jpg", cv::IMREAD_GRAYSCALE);
	
	/*cv::Mat edgesx, edgesy;
	cv::Scharr(img, edgesx, -1, 0, 1);
	cv::Scharr(img, edgesy, -1, 1, 0);
	img = (img + (edgesx + edgesy) / 2) / 2;*/

	//imshow("Image", img);
	/*float wavelengthMin = 4 / sqrt(2);
	float wavelengthMax = hypot(img.rows, img.cols);
	int n = floor(log2(wavelengthMax / wavelengthMin));
	int dTheta = 45;
	float wavelength, sigma;
	int size;
	cv::Mat kernel;
	cv::Mat blurredImg;
	vector<cv::Mat> filteredImgs;
	double minv, maxv;

	float freqs[] = { 23, 31, 47 };

	for (int orientation = 0; orientation <= 180 - dTheta; orientation += dTheta) {
		for (int p = 0; p <= n-2; p++) {
		//for(int f = 0; f <= 2; f++) {
			wavelength = pow(2, p) * wavelengthMin;
			//wavelength = hypot(img.rows, img.cols) / freqs[f];
			sigma = 0.56 * wavelength;
			size = 2 * ceil(7 * 2 * sigma) + 1;

			//GaussianBlur(img, blurredImg, cv::Size(size, size), sigma, 2 * sigma);

			//real part
			kernel = cv::getGaborKernel(cv::Size(size, size), sigma, orientation, wavelength, 0.5, 0, CV_32F);
			cv::Mat realImg(img.size(), CV_32F);
			cv::filter2D(img, realImg, CV_32F, kernel);
			//cv::divide(realImg, blurredImg, realImg, 1, CV_32F);

			//imaginary part
			kernel = cv::getGaborKernel(cv::Size(size, size), sigma, orientation, wavelength, 0.5, CV_PI, CV_32F);
			cv::Mat imagImg(img.size(), CV_32F);
			cv::filter2D(img, imagImg, CV_32F, kernel);
			//cv::divide(imagImg, blurredImg, imagImg, 1, CV_32F);

			//magnitude (Gabor energy)
			/*cv::Mat magImg(img.size(), CV_32F);
			cv::pow(realImg, 2, realImg);
			cv::pow(imagImg, 2, imagImg);
			cv::add(realImg, imagImg, magImg);
			cv::sqrt(magImg, magImg);

			size = 2 * ceil(2 * sigma * 3) + 1;
			GaussianBlur(magImg, magImg, cv::Size(size, size), sigma * 3);

			filteredImgs.push_back(magImg);*/

			/*size = 2 * ceil(2 * sigma * 2) + 1;
			GaussianBlur(realImg, realImg, cv::Size(size, size), sigma * 2);
			GaussianBlur(imagImg, imagImg, cv::Size(size, size), sigma * 2);
			filteredImgs.push_back(realImg);
			filteredImgs.push_back(imagImg);*/

			/*if (orientation == 0) {
				cv::minMaxIdx(magImg, &minv, &maxv);
				cv::convertScaleAbs(magImg, magImg, 255 / maxv);
				imshow("gabor" + to_string(orientation) + "_" + to_string(p), magImg);
				//imshow("gabor" + to_string(orientation) + "_" + to_string(p), kernel);
			}*/
		//}
	//}

	/*int gaborsCnt = filteredImgs.size();
	filteredImgs.push_back(cv::Mat(img.size(), CV_32F));
	filteredImgs.push_back(cv::Mat(img.size(), CV_32F));
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			filteredImgs[gaborsCnt].at<float>(x, y) = x;
			filteredImgs[gaborsCnt+1].at<float>(x, y) = y;
		}
	}*/

	/*cv::Mat gabors, meanG, stdG, imgData;
	for (cv::Mat m : filteredImgs) {
		meanStdDev(m, meanG, stdG);
		m -= meanG;
		m /= stdG;	
		imgData = cv::Mat(1, img.rows * img.cols, CV_32F, m.data);
		gabors.push_back(imgData);
	}

	cv::PCA pca(gabors, cv::noArray(), CV_PCA_DATA_AS_COL, 1);
	cv::Mat components = pca.project(gabors);
	//cv::reduce(mainComponent, mainComponent, 0, CV_REDUCE_SUM);
	cv::Mat mainComponent = components.row(0);
	mainComponent = mainComponent.reshape(1, img.rows);

	cv::minMaxIdx(mainComponent, &minv, &maxv);
	mainComponent -= minv;
	cv::convertScaleAbs(mainComponent, mainComponent, 255 / (maxv - minv));
	imshow("PCA", mainComponent);


	int k = 6;

	cv::Mat labels, centers;
	cv::kmeans(gabors.t(), k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		3, KMEANS_PP_CENTERS, centers);
	labels = labels.reshape(1, img.rows);
	labels += 1;
	cv::Mat labs;
	cv::convertScaleAbs(labels, labs, 255 / k);
	imshow("labels", labs);

	/*cv::Mat mask;
	for (int i = 1; i <= k; i++) {
	cv::inRange(labels, i, i, mask);
	//cv::minMaxIdx(mask, &minv, &maxv);
	//cout << minv << " " << maxv << endl;
	//imshow("mask"+to_string(i), mask );
	if (cv::countNonZero(mask) < 10000) {
	labels.setTo(0, mask);
	}
	}

	cv::convertScaleAbs(labels, labels, 255 / k);
	imshow("labels after removal", labels);

	labels.convertTo(labels, CV_32SC1);
	cv::watershed(cv::Mat(img.size(), CV_8UC3), labels);
	cv::minMaxIdx(labels, &minv, &maxv);
	labels -= minv;
	cv::convertScaleAbs(labels, labels, 255 / (maxv - minv));
	imshow("Watershed", labels);*/

	/*float step = 255.0 / float(k);
	cv::Mat mask, segments = cv::Mat::zeros(img.size(), CV_8U);
	for (int i = 0; i < k; i++) {
	cv::inRange(mainComponent, i * step, (i + 1) * step, mask);
	segments.setTo((i + 1) * step, mask);
	}
	imshow("Segments", segments);*/


	/*size = 3;
	cv::Mat cimg = cv::imread("./database/set_classify/img.jpg", cv::IMREAD_COLOR);

	vector<cv::Mat> hsvImgs(3);
	split(cimg, hsvImgs);
	cv::Mat hue = hsvImgs[0];
	GaussianBlur(hue, hue, cv::Size(size, size), 0);
	imshow("hue", hue);

	cv::Mat hueReshape = hue.reshape(1, hue.rows * hue.cols);
	hueReshape.convertTo(hueReshape, CV_32F);

	pca = cv::PCA(hueReshape, cv::noArray(), CV_PCA_DATA_AS_ROW, 1);
	components = pca.project(hueReshape);
	mainComponent = components.col(0);
	mainComponent = mainComponent.reshape(1, img.rows);
	cv::minMaxIdx(mainComponent, &minv, &maxv);
	mainComponent -= minv;
	cv::convertScaleAbs(mainComponent, mainComponent, 255 / (maxv - minv));
	imshow("hue PCA", mainComponent);

	cv::kmeans(hueReshape, k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		3, KMEANS_PP_CENTERS, centers);
	labels = labels.reshape(1, hue.rows);
	labels += 1;
	cv::convertScaleAbs(labels, labels, 255 / k);
	imshow("hue labels", labels);


	cv::merge(hsvImgs, cimg);
	GaussianBlur(cimg, cimg, cv::Size(size, size), 0);
	imshow("cimg", cimg);

	cv::Mat cimgReshape = cimg.reshape(1, cimg.rows * cimg.cols);
	cimgReshape.convertTo(cimgReshape, CV_32F);

	pca = cv::PCA(cimgReshape, cv::noArray(), CV_PCA_DATA_AS_ROW, 1);
	components = pca.project(cimgReshape);
	mainComponent = components.col(0);
	mainComponent = mainComponent.reshape(1, img.rows);
	cv::minMaxIdx(mainComponent, &minv, &maxv);
	mainComponent -= minv;
	cv::convertScaleAbs(mainComponent, mainComponent, 255 / (maxv - minv));
	imshow("color PCA", mainComponent);

	cv::kmeans(cimgReshape, k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		3, KMEANS_PP_CENTERS, centers);
	labels = labels.reshape(1, cimg.rows);
	labels += 1;
	cv::convertScaleAbs(labels, labels, 255 / k);
	imshow("color labels", labels);


	GaussianBlur(img, img, cv::Size(size, size), 0);
	imshow("img", img);
	cv::Mat imgReshape = img.reshape(1, img.rows * img.cols);
	imgReshape.convertTo(imgReshape, CV_32F);

	pca = cv::PCA(imgReshape, cv::noArray(), CV_PCA_DATA_AS_ROW, 1);
	components = pca.project(imgReshape);
	mainComponent = components.col(0);
	mainComponent = mainComponent.reshape(1, img.rows);
	cv::minMaxIdx(mainComponent, &minv, &maxv);
	mainComponent -= minv;
	cv::convertScaleAbs(mainComponent, mainComponent, 255 / (maxv - minv));
	imshow("value PCA", mainComponent);

	cv::kmeans(imgReshape, k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		3, KMEANS_PP_CENTERS, centers);
	labels = labels.reshape(1, img.rows);
	labels += 1;
	cv::convertScaleAbs(labels, labels, 255 / k);
	imshow("value labels", labels);*/


	waitKey(0); // Wait for a keystroke in the window
	getchar();
	return 0;
}