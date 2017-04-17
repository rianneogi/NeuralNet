#include "UnitError.h"

UnitError::UnitError() : ErrorFunction()
{
}

UnitError::UnitError(Blob* input, Blob* output) : ErrorFunction(input, output, new Tensor(output->Data.mShape))
{
}

UnitError::UnitError(Blob* input, Blob* output, Tensor* target) : ErrorFunction(input, output, target)
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
		error += abs(mOutput->Data(i) - (*mTarget)(i));
		printf("err %f, %f, %f\n", abs(mOutput->Data(i) - (*mTarget)(i)), mOutput->Data(i), (*mTarget)(i));
		mOutput->Delta(i) += 1.0;
	}

	return error;
}

void UnitError::backprop()
{
}