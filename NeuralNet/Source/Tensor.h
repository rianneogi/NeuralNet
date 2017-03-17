#pragma once

#include "UtilFuncs.h"

extern cl_context gCLContext;
extern cl_command_queue gCLQueue;

typedef std::vector<uint64_t> TensorShape;

class Tensor
{
public:
	TensorShape mShape;
	uint64_t mSize;
	Float* mData;
	cl_mem mMemory;

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

	void allocateCPU();
	void allocateGPU();

	void freemem();
	void freeCPU();
	void freeGPU();

	void copyToGPU();
	void copyToCPU();

	void setzero();
	void setconstant(Float c);
	void setidentity();

	//Tensor subtensor(const TensorShape& begin, const TensorShape& size);
	Tensor cut(uint64_t begin, uint64_t len) const; //cuts the tensor based on primary dimension

	uint64_t rows() const;
	uint64_t cols() const;

	void print() const;
	void printshape() const;
};

TensorShape make_shape(uint64_t a);
TensorShape make_shape(uint64_t a, uint64_t b);
TensorShape make_shape(uint64_t a, uint64_t b, uint64_t c);
TensorShape make_shape(uint64_t a, uint64_t b, uint64_t c, uint64_t d);

inline void gemm_cpu(Tensor* m1, Tensor* m2, Tensor* res, CBLAS_TRANSPOSE trans_m1, CBLAS_TRANSPOSE trans_m2, Float alpha, Float beta)
{
#ifdef NN_DEBUG
	uint64_t M = trans_m1 == CblasNoTrans ? m1->rows() : m1->cols();
	uint64_t N = trans_m2 == CblasNoTrans ? m2->cols() : m2->rows();
	uint64_t K = trans_m1 == CblasNoTrans ? m1->cols() : m1->rows();
	uint64_t L = trans_m2 == CblasNoTrans ? m2->rows() : m2->cols();
	assert(K == L);
	assert(M == res->rows());
	assert(N == res->cols());
#endif
	cblas_sgemm(CblasRowMajor, trans_m1, trans_m2,
		res->rows(), //M
		res->cols(), //N
		trans_m1 == CblasNoTrans ? m1->cols() : m1->rows(), //K
		alpha,
		m1->mData, m1->cols(),
		m2->mData, m2->cols(),
		beta,
		res->mData, res->cols());
}
