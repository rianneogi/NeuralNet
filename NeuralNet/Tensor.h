#pragma once

#include "RecurrentNeuralNet.h"
#include <unsupported\Eigen\CXX11\src\Tensor\TensorMap.h>

class Tensor
{
public:
	std::vector<unsigned int> mDims;
	Eigen::Tensor<float, 3, Eigen::RowMajor, Eigen::DenseIndex> t;
	//Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned, false> tm;

	Tensor();
	Tensor(unsigned int a);
	Tensor(unsigned int a, unsigned int b);
	Tensor(unsigned int a, unsigned int b, unsigned int c);
	Tensor(unsigned int a, unsigned int b, unsigned int c, unsigned int d);
	~Tensor();
};