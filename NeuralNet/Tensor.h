#pragma once

#include "RecurrentNeuralNet.h"
//#include <unsupported\Eigen\CXX11\src\Tensor\TensorMap.h>

typedef std::vector<unsigned int> TensorShape;

class Tensor
{
public:
	std::vector<unsigned int> mShape;
	Float* mData;

	Tensor();
	Tensor(const TensorShape& shape);
	~Tensor();

	Float& operator()(unsigned int a);
	Float& operator()(unsigned int a, unsigned int b);
	Float& operator()(unsigned int a, unsigned int b, unsigned int c);
	Float& operator()(unsigned int a, unsigned int b, unsigned int c, unsigned int d);

	void allocate();
	void free();

	unsigned int rows();
	unsigned int cols();

	void print();
};

TensorShape make_shape(unsigned int a);
TensorShape make_shape(unsigned int a, unsigned int b);
TensorShape make_shape(unsigned int a, unsigned int b, unsigned int c);
TensorShape make_shape(unsigned int a, unsigned int b, unsigned int c, unsigned int d);