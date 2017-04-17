#include "ErrorFunction.h"

ErrorFunction::ErrorFunction()
{
}

ErrorFunction::ErrorFunction(Blob* input, Blob* output, Tensor* target) : mInput(input), mOutput(output), mTarget(target)
{
}

ErrorFunction::~ErrorFunction()
{
}
