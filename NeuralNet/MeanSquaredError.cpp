#include "MeanSquaredError.h"

MeanSquaredError::MeanSquaredError() : ErrorFunction()
{
}

MeanSquaredError::MeanSquaredError(Matrix* input, Matrix* output, Matrix* target) : ErrorFunction(input, output, target)
{
}

MeanSquaredError::~MeanSquaredError()
{
}

Float MeanSquaredError::calculateError()
{
	//return ((*mTarget - *mOutput).unaryExpr(square)).sum();
	return 0.0;
}
