#pragma once

#include "UtilFuncs.h"
//#include <unsupported\Eigen\CXX11\src\Tensor\TensorMap.h>

typedef std::vector<unsigned int> TensorShape;

class Tensor
{
public:
	std::vector<unsigned int> mShape;
	unsigned int mSize;
	Float* mData;

	Tensor();
	Tensor(const TensorShape& shape);
	~Tensor();

	Float& operator()(unsigned int a);
	Float& operator()(unsigned int a, unsigned int b);
	Float& operator()(unsigned int a, unsigned int b, unsigned int c);
	Float& operator()(unsigned int a, unsigned int b, unsigned int c, unsigned int d);

	void allocate();
	void freememory();

	void setzero();
	void setidentity();

	unsigned int rows();
	unsigned int cols();

	void print();
};

TensorShape make_shape(unsigned int a);
TensorShape make_shape(unsigned int a, unsigned int b);
TensorShape make_shape(unsigned int a, unsigned int b, unsigned int c);
TensorShape make_shape(unsigned int a, unsigned int b, unsigned int c, unsigned int d);

inline void matmul(Tensor* m1, Tensor* m2, Tensor* res)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1->cols(), m2->rows(),
		m1->rows(), 1, m1->mData, m1->rows(), m2->mData, m2->rows(), 0, res->mData, res->rows());
}