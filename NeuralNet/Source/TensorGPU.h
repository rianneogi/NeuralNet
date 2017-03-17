#pragma once

#include "Tensor.h"

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
	TensorGPU(Float* data, const TensorShape& shape); //initialize tensor pointing to existing data
	~TensorGPU();

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
	TensorGPU cut(uint64_t begin, uint64_t len) const; //cuts the tensor based on primary dimension

	uint64_t rows() const;
	uint64_t cols() const;

	void print() const;
	void printshape() const;
};
