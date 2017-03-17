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

	void reshape(const TensorShape& shape);
};
