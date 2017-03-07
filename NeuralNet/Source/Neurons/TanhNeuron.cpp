#include "TanhNeuron.h"

TanhNeuron::TanhNeuron() : Neuron()
{
}

TanhNeuron::TanhNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output, learning_rate)
{
	assert(input->Data.cols() == output->Data.cols() && input->Data.rows() == output->Data.rows());
}

TanhNeuron::~TanhNeuron()
{
}

void TanhNeuron::forward()
{
	/*mOutput->Data = mInput->Data.unaryExpr(&tanh_NN);*/
}

void TanhNeuron::backprop()
{
	//mInput->Delta = mOutput->Delta.cwiseProduct(Matrix::Constant(mOutput->Data.rows(), mOutput->Data.cols(), 1.0) - mOutput->Data.cwiseProduct(mOutput->Data));

	////Weights = Weights - (mLearningRate*mOutput->Delta*mInput->Data.transpose());
	////assert(Weights.cols() == InputSize && Weights.rows() == OutputSize);

	////Biases = Biases - (mLearningRate*mOutput->Delta*Matrix::Constant(mOutput->Delta.cols(), 1, 1.0));
}
