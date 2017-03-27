#include "MeanSquaredError.h"

MeanSquaredError::MeanSquaredError() : ErrorFunction()
{
}

MeanSquaredError::MeanSquaredError(Blob* input, Blob* output) : ErrorFunction(input, output, nullptr)
{
}

MeanSquaredError::MeanSquaredError(Blob* input, Blob* output, const Tensor* target) : ErrorFunction(input, output, target)
{
}

MeanSquaredError::~MeanSquaredError()
{
}

Float MeanSquaredError::calculateError()
{
	if (mTarget == nullptr) 
		return 0;

	Float error = 0;
	for (int i = 0; i < mOutput->Data.mSize; i++)
	{
		error += 0.5*(mOutput->Data(i) - (*mTarget)(i))*(mOutput->Data(i) - (*mTarget)(i));
		mOutput->Delta(i) += mOutput->Data(i) - (*mTarget)(i);
	}
	
	return error;
}

void MeanSquaredError::backprop()
{
}
