#pragma once

#include "ErrorFunction.h"

class MeanSquaredError : public ErrorFunction
{
public:
	MeanSquaredError();
	MeanSquaredError(Matrix* input, Matrix* output, Matrix* target);
	~MeanSquaredError();

	Float calculateError();
};

