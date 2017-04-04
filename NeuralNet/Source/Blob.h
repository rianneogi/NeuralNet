#pragma once

#include "TensorGPU.h"

class Blob
{
public:
	Tensor Data;
	Tensor Delta;

	Blob();
	Blob(const TensorShape& shape);
	Blob(Tensor data, Tensor delta);
	~Blob();

	void copyToGPU();
	void copyToCPU();

	void reshape(const TensorShape& shape);

	Blob* cut(uint64_t start, uint64_t len);
};
