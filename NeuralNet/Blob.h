#pragma once

#include "Tensor.h"

class Blob
{
public:
	Matrix Data;
	Matrix Delta;

	Blob();
	Blob(unsigned int rows, unsigned int cols);
	~Blob();
};
