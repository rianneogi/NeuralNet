#include "L1Error.h"

L1Error::L1Error() : ErrorFunction()
{
}

L1Error::L1Error(Blob* input, Blob* output) : ErrorFunction(input, output, nullptr)
{
}

L1Error::L1Error(Blob* input, Blob* output, const Tensor* target) : ErrorFunction(input, output, target)
{
}

L1Error::~L1Error()
{
}

Float L1Error::calculateError()
{
	if (mTarget == nullptr)
		return 0;

	Float error = 0;
	for (int i = 0; i < mOutput->Data.mSize; i++)
	{
		error += abs(mOutput->Data(i) - (*mTarget)(i));
		mOutput->Delta(i) += mOutput->Data(i) > (*mTarget)(i)? 1.0 : (mOutput->Data(i) == (*mTarget)(i) ? 0.0 : -1.0);
	}

	return error;
}

void L1Error::backprop()
{
}
