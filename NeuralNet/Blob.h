#pragma once

#include "Tensor.h"

class Blob
{
public:
	Tensor Data;
	Tensor Delta;

	Blob();
	Blob(unsigned int rows, unsigned int cols);
	~Blob();
};
