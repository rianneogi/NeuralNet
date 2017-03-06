#pragma once

#include "../ErrorFunction.h"

class MeanSquaredError : public ErrorFunction
{
public:
	MeanSquaredError();
	MeanSquaredError(Blob* input, Blob* output, const Matrix* target);
	~MeanSquaredError();

	Float calculateError();
	void backprop();
};

