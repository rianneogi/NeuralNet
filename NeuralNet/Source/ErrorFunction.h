#pragma once

#include "Neurons/SigmoidNeuron.h"

class ErrorFunction
{
public:
	Blob* mInput;
	Blob* mOutput;
	Tensor* mTarget;

	ErrorFunction();
	ErrorFunction(Blob* input, Blob* output, Tensor* target);
	~ErrorFunction();

	virtual Float calculateError() = 0;
	virtual void backprop() = 0;
};

