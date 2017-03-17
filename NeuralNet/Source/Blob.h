#pragma once

#include "TensorGPU.h"

class Blob
{
public:
	Tensor Data;
	Tensor Delta;

	Blob();
	Blob(const TensorShape& shape);
	~Blob();

	void copyToGPU();
	void copyToCPU();

	void reshape(const TensorShape& shape);
};
