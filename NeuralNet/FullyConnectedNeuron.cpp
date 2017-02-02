#include "FullyConnectedNeuron.h"

FullyConnectedNeuron::FullyConnectedNeuron()
{
}

FullyConnectedNeuron::FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output, learning_rate)
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

FullyConnectedNeuron::~FullyConnectedNeuron()
{
}

void FullyConnectedNeuron::forward()
{
	mOutput->Data = (Weights * mInput->Data) + (Biases.replicate(1, (mInput->Data).cols()));
}

void FullyConnectedNeuron::backprop()
{
	//assert(mOutput->Delta.cols() == BatchSize && mOutput->Delta.rows() == OutputSize);
	mInput->Delta = Weights.transpose()*mOutput->Delta;
	//assert(mInput->Delta.cols() == BatchSize && mInput->Delta.rows() == InputSize);

	Weights = Weights - (mLearningRate*mOutput->Delta*mInput->Data.transpose());
	assert(Weights.cols() == InputSize && Weights.rows() == OutputSize);

	//Vector DeltaSum(mOutput->Delta.rows());
	//for (int j = 0; j < DeltaSum.size(); j++)
	//	DeltaSum[j] = (mOutput->Delta.row(j)).sum();

	Biases = Biases - (mLearningRate*mOutput->Delta*Matrix::Constant(mOutput->Delta.cols(), 1, 1.0));
}
