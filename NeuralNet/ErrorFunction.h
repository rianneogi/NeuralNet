#pragma once

#include "SigmoidNeuron.h"

class ErrorFunction
{
public:
	Matrix* mInput;
	Matrix* mOutput;
	Matrix* mTarget;

	ErrorFunction();
	ErrorFunction(Matrix* input, Matrix* output, Matrix* target);
	~ErrorFunction();

	virtual Float calculateError() = 0;
};

