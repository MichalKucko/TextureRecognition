#include "SimpleBlur.h"
#include "opencv2/imgproc.hpp"

void SimpleBlur::preprocess(cv::Mat &src, cv::Mat &dst) {
	blur(src, dst, cv::Size(size, size));
}