#include "MeanSquaredError.h"

MeanSquaredError::MeanSquaredError() : ErrorFunction()
{
}

MeanSquaredError::MeanSquaredError(Blob* input, Blob* output, const Matrix* target) : ErrorFunction(input, output, target)
{
}

MeanSquaredError::~MeanSquaredError()
{
}

Float MeanSquaredError::calculateError()
{
	double error = 0.5*((mOutput->Data - *mTarget).cwiseProduct(mOutput->Data - *mTarget)).sum();
	mOutput->Delta = (mOutput->Data - *mTarget);
	return error;
	//return 0.0;
}

void MeanSquaredError::backprop()
{
}
