#pragma once

#include "UtilFuncs.h"
//#include <unsupported\Eigen\CXX11\src\Tensor\TensorMap.h>

typedef std::vector<uint64_t> TensorShape;

class Tensor
{
public:
	TensorShape mShape;
	uint64_t mSize;
	Float* mData;
	bool mSelfAllocated;

	Tensor();
	Tensor(const Tensor& other);
	Tensor(const TensorShape& shape); //initialize tensor allocated with given shape
	Tensor(Float* data, const TensorShape& shape); //initialize tensor pointing to existing data
	~Tensor();

	//void operator=(const Tensor& other);

	Float& operator()(uint64_t a) const;
	Float& operator()(uint64_t a, uint64_t b) const;
	Float& operator()(uint64_t a, uint64_t b, uint64_t c) const;
	Float& operator()(uint64_t a, uint64_t b, uint64_t c, uint64_t d) const;

	void allocate();
	void freememory();

	void setzero();
	void setconstant(Float c);
	void setidentity();

	//Tensor subtensor(const TensorShape& begin, const TensorShape& size);
	Tensor cut(uint64_t begin, uint64_t len) const; //cuts the tensor based on primary dimension

	uint64_t rows() const;
	uint64_t cols() const;

	void print() const;
};

TensorShape make_shape(uint64_t a);
TensorShape make_shape(uint64_t a, uint64_t b);
TensorShape make_shape(uint64_t a, uint64_t b, uint64_t c);
TensorShape make_shape(uint64_t a, uint64_t b, uint64_t c, uint64_t d);

inline void gemm(Tensor* m1, Tensor* m2, Tensor* res, CBLAS_TRANSPOSE trans_m1, CBLAS_TRANSPOSE trans_m2, Float alpha, Float beta)
{
	uint64_t M = trans_m1 == CblasNoTrans ? m1->rows() : m1->cols();
	uint64_t N = trans_m2 == CblasNoTrans ? m2->cols() : m2->rows();
	uint64_t K = trans_m1 == CblasNoTrans ? m1->cols() : m1->rows();
	uint64_t L = trans_m2 == CblasNoTrans ? m2->rows() : m2->cols();
	assert(K == L);
	assert(M == res->rows());
	assert(N == res->cols());
	cblas_dgemm(CblasRowMajor, trans_m1, trans_m2,
		res->rows(), //M
		res->cols(), //N
		trans_m1 == CblasNoTrans ? m1->cols() : m1->rows(), //K
		alpha, 
		m1->mData, m1->cols(),
		m2->mData, m2->cols(),
		beta, 
		res->mData, res->cols());
}

inline void gemm_conv(Tensor* m1, Tensor* m2, Tensor* res, CBLAS_TRANSPOSE trans_m1, CBLAS_TRANSPOSE trans_m2, Float alpha, Float beta, 
	unsigned int conv_size)
{
	/*unsigned int M = trans_m1 == CblasNoTrans ? m1->rows() : m1->cols();
	unsigned int N = trans_m2 == CblasNoTrans ? m2->rows() : m2->cols();
	unsigned int K = trans_m1 == CblasNoTrans ? m1->cols() : m1->rows();*/
	cblas_dgemm(CblasRowMajor, trans_m1, trans_m2,
		res->rows() * conv_size, //M
		res->cols() / conv_size, //N
		trans_m1 == CblasNoTrans ? m1->cols() : m1->rows(), //K
		alpha,
		m1->mData, m1->cols(),
		m2->mData, m2->cols(),
		beta,
		res->mData, res->cols() / conv_size);
}