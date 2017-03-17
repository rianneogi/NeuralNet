#pragma once

#include "Tensor.h"

extern cl_context gCLContext;
extern cl_command_queue gCLQueue;

class TensorGPU
{
public:
	TensorShape mShape;
	uint64_t mSize;
	cl_float* mData;
	cl_mem mMemory;

	TensorGPU();
	TensorGPU(const TensorGPU& other);
	TensorGPU(const TensorShape& shape); //initialize tensor allocated with given shape
	TensorGPU(cl_float* data, const TensorShape& shape); //initialize tensor pointing to existing data
	~TensorGPU();

	//void operator=(const Tensor& other);

	cl_float& operator()(uint64_t a) const;
	cl_float& operator()(uint64_t a, uint64_t b) const;
	cl_float& operator()(uint64_t a, uint64_t b, uint64_t c) const;
	cl_float& operator()(uint64_t a, uint64_t b, uint64_t c, uint64_t d) const;

	void allocate();
	void freememory();

	void setzero();
	void setconstant(cl_float c);
	void setidentity();

	//Tensor subtensor(const TensorShape& begin, const TensorShape& size);
	TensorGPU cut(uint64_t begin, uint64_t len) const; //cuts the tensor based on primary dimension

	void copyToGPU();
	void copyToCPU();

	uint64_t rows() const;
	uint64_t cols() const;

	void print() const;
	void printshape() const;
};

void cl_error(cl_int err);

inline void gemm_gpu(TensorGPU* m1, TensorGPU* m2, TensorGPU* res, clblasTranspose trans_m1, clblasTranspose trans_m2, Float alpha, Float beta)
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
	cl_event event = NULL;
	cl_int err = clblasSgemm(clblasRowMajor, trans_m1, trans_m2,
		res->rows()-1, //M
		res->cols()-1, //N
		trans_m1 == clblasNoTrans ? m1->cols()-1 : m1->rows()-1, //K
		alpha,
		m1->mMemory, m1->cols()+1, m1->cols(),
		m2->mMemory, m2->cols()+1, m2->cols(),
		beta,
		res->mMemory, res->cols()+1, res->cols(),
		1, &gCLQueue, 0, NULL, &event);
	if (err != CL_SUCCESS)
	{
		printf("ERROR: sgemm: %d\n", err);
		cl_error(err);
	}
	printf("%d\n", event);
	err = clWaitForEvents(1, &event); 
	if (err != CL_SUCCESS)
	{
		printf("ERROR: gemm gpu: %d\n", err);
		cl_error(err);
	}
}