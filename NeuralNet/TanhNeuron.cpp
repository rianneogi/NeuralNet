#include "TanhNeuron.h"

TanhNeuron::TanhNeuron()
{
}

TanhNeuron::TanhNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output, learning_rate)
{
	Weights = Matrix(output->Data.rows(), input->Data.rows());
	Biases = Vector(output->Data.rows());
	for (int i = 0; i < Weights.rows(); i++)
	{
		Biases[i] = rand_init();
		for (int j = 0; j < Weights.cols(); j++)
		{
			Weights(i, j) = rand_init();
		}
	}
	InputSize = Weights.cols();
	OutputSize = Weights.rows();
	BatchSize = output->Data.cols();
	assert(input->Data.cols() == output->Data.cols());
}

TanhNeuron::~TanhNeuron()
{
}

void TanhNeuron::forward()
{
	mOutput->Data = ((Weights * mInput->Data) + (Biases.replicate(1, (mInput->Data).cols()))).unaryExpr(&tanh_NN);
}

void TanhNeuron::backprop()
{
	mInput->Delta = (Weights.transpose()*mOutput->Delta).cwiseProduct(Matrix::Constant(mInput->Data.rows(), mInput->Data.cols(), 1.0) - mInput->Data.cwiseProduct(mInput->Data));

	Weights = Weights - (mLearningRate*mOutput->Delta*mInput->Data.transpose());
	assert(Weights.cols() == InputSize && Weights.rows() == OutputSize);

	Biases = Biases - (mLearningRate*mOutput->Delta*Matrix::Constant(mOutput->Delta.cols(), 1, 1.0));
}
