#include "MeanSquaredError.h"

MeanSquaredError::MeanSquaredError() : ErrorFunction()
{
}

MeanSquaredError::MeanSquaredError(Blob* input, Blob* output, Tensor* target) : ErrorFunction(input, output, target)
{
}

MeanSquaredError::~MeanSquaredError()
{
}

Float MeanSquaredError::calculateError()
{
	//double error = 0.5*((mOutput->Data - *mTarget).cwiseProduct(mOutput->Data - *mTarget)).sum();
	//mOutput->Delta = (mOutput->Data - *mTarget);
	//return error;

	Float error = 0;
	for (int i = 0; i < mOutput->Data.mSize; i++)
	{
		error += 0.5*(mOutput->Data(i) - (*mTarget)(i))*(mOutput->Data(i) - (*mTarget)(i));
		mOutput->Delta(i) = mOutput->Data(i) - (*mTarget)(i);
	}
}

void MeanSquaredError::backprop()
{
}
