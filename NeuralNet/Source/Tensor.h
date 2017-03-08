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
	bool mSelfAllocated;

	Tensor();
	Tensor(const TensorShape& shape); //initialize tensor allocated with given shape
	Tensor(Float* data, const TensorShape& shape); //initialize tensor pointing to existing data
	~Tensor();

	Float& operator()(unsigned int a) const;
	Float& operator()(unsigned int a, unsigned int b) const;
	Float& operator()(unsigned int a, unsigned int b, unsigned int c) const;
	Float& operator()(unsigned int a, unsigned int b, unsigned int c, unsigned int d) const;

	void allocate();
	void freememory();

	void setzero();
	void setidentity();

	//Tensor subtensor(const TensorShape& begin, const TensorShape& size);
	Tensor cut(unsigned int begin, unsigned int len) const; //cuts the tensor based on primary dimension

	unsigned int rows() const;
	unsigned int cols() const;

	void print() const;
};

TensorShape make_shape(unsigned int a);
TensorShape make_shape(unsigned int a, unsigned int b);
TensorShape make_shape(unsigned int a, unsigned int b, unsigned int c);
TensorShape make_shape(unsigned int a, unsigned int b, unsigned int c, unsigned int d);

inline void gemm(Tensor* m1, Tensor* m2, Tensor* res, CBLAS_TRANSPOSE trans_m1, CBLAS_TRANSPOSE trans_m2, Float alpha, Float beta)
{
	cblas_dgemm(CblasRowMajor, trans_m1, trans_m2, m1->rows(), m2->cols(), m1->cols(), alpha, m1->mData,
		trans_m1 == CblasTrans? m1->rows(): m1->cols(), m2->mData, trans_m2 == CblasTrans? m2->rows(): m2->cols(),
		beta, res->mData, res->cols());
}