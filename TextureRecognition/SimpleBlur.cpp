#include "SimpleBlur.h"
#include "opencv2/imgproc.hpp"

void SimpleBlur::preprocess(cv::Mat &img) {
	blur(img, img, cv::Size(size, size));
}