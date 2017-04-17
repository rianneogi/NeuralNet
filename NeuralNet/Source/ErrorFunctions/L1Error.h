#pragma once

#include "../ErrorFunction.h"

class L1Error : public ErrorFunction
{
public:
	L1Error();
	L1Error(Blob* input, Blob* output);
	L1Error(Blob* input, Blob* output, Tensor* target);
	~L1Error();

	Float calculateError();
	void backprop();
};
