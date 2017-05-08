#pragma once
#include "Preprocessing.h"


/**
* Applies simple mean filter to the given image.
*/
class SimpleBlur : public Preprocessing {
private:

	/**
	* filter size
	*/
	int size;

public:

	SimpleBlur() : size(3) {};

	/**
	  \param size - filter size 
	*/
	SimpleBlur(int _size) : size(_size) {};

	/**
	* Sets filter size.
	  \param size - filter size 
	*/
	void setSize(int _size) { size = _size; };

	void preprocess(cv::Mat &src, cv::Mat &dst);
};