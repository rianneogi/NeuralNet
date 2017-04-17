#pragma once

#include "../ErrorFunction.h"

//Error Function with Delta = 1

class UnitError : public ErrorFunction
{
public:
	UnitError();
	UnitError(Blob* input, Blob* output);
	UnitError(Blob* input, Blob* output, Tensor* target);
	~UnitError();

	Float calculateError();
	void backprop();
};