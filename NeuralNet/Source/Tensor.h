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

inline void gemm(Tensor* m1, Tensor* m2, Tensor* res, CBLAS_TRANSPOSE trans_m1, CBLAS_TRANSPOSE trans_m2, Float alpha, Float beta)
{
	cblas_dgemm(CblasRowMajor, trans_m1, trans_m2, m1->cols(), m2->rows(), m1->rows(), alpha, m1->mData,
		trans_m1 == CblasTrans? m1->cols: m1->rows(), m2->mData, trans_m2 == CblasTrans? m2->cols(): m2->rows(),
		beta, res->mData, res->rows());
}