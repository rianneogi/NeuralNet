#include "TanhNeuron.h"

TanhNeuron::TanhNeuron() : Neuron()
{
}

TanhNeuron::TanhNeuron(Blob* input, Blob* output) : Neuron(input, output)
{
	assert(input->Data.mSize == output->Data.mSize);
}

TanhNeuron::~TanhNeuron()
{
}

void TanhNeuron::forward()
{
	/*mOutput->Data = mInput->Data.unaryExpr(&tanh_NN);*/
	for (unsigned int i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = tanh(mInput->Data(i));
	}
}

void TanhNeuron::backprop()
{
	//mInput->Delta = mOutput->Delta.cwiseProduct(Matrix::Constant(mOutput->Data.rows(), mOutput->Data.cols(), 1.0) - mOutput->Data.cwiseProduct(mOutput->Data));

	////Weights = Weights - (mLearningRate*mOutput->Delta*mInput->Data.transpose());
	////assert(Weights.cols() == InputSize && Weights.rows() == OutputSize);

	////Biases = Biases - (mLearningRate*mOutput->Delta*Matrix::Constant(mOutput->Delta.cols(), 1, 1.0));

	for (int i = 0; i < mInput->Delta.mSize; i++)
	{
		mInput->Delta(i) = mOutput->Delta(i)*(1.0 - mOutput->Data(i)*mOutput->Data(i));
	}
}

std::vector<Blob*> TanhNeuron::getVariables()
{
	return std::vector<Blob*>();
}
