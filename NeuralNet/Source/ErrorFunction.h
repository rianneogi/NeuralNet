#pragma once

#include "Neurons/SigmoidNeuron.h"

class ErrorFunction
{
public:
	Blob* mInput;
	Blob* mOutput;
	const Matrix* mTarget;

	ErrorFunction();
	ErrorFunction(Blob* input, Blob* output, const Matrix* target);
	~ErrorFunction();

	virtual Float calculateError() = 0;
	virtual void backprop() = 0;
};

