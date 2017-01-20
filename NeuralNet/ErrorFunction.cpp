#include "ErrorFunction.h"

ErrorFunction::ErrorFunction()
{
}

ErrorFunction::ErrorFunction(Matrix* input, Matrix* output, Matrix* target) : mInput(input), mOutput(output), mTarget(target)
{
}

ErrorFunction::~ErrorFunction()
{
}
