#include "UnitError.h"

UnitError::UnitError() : ErrorFunction()
{
}

UnitError::UnitError(Blob* input, Blob* output) : ErrorFunction(input, output, nullptr)
{
}

UnitError::UnitError(Blob* input, Blob* output, const Tensor* target) : ErrorFunction(input, output, target)
{
}

UnitError::~UnitError()
{
}

Float UnitError::calculateError()
{
	if (mTarget == nullptr)
		return 0;

	Float error = 0;
	for (int i = 0; i < mOutput->Data.mSize; i++)
	{
		error += (mOutput->Data(i) - (*mTarget)(i));
		mOutput->Delta(i) += 1.0;
	}

	return error;
}

void UnitError::backprop()
{
}