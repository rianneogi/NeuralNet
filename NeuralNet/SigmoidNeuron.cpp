#include "SigmoidNeuron.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

SigmoidNeuron::SigmoidNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output, learning_rate)
{
	for (int i = 0; i < Weights.rows(); i++)
	{
		Biases[i] = rand_init();
		for (int j = 0; j < Weights.cols(); j++)
		{
			Weights(i, j) = rand_init();
		}
	}
}

SigmoidNeuron::~SigmoidNeuron()
{
}

void SigmoidNeuron::forward()
{
	mOutput->Data = ((Weights * mInput->Data) + (Biases.replicate(1, (mInput->Data).cols()))).unaryExpr(&sigmoid);
}

void SigmoidNeuron::backprop()
{
	mInput->Delta = (Weights.transpose()*mOutput->Delta).cwiseProduct(mOutput->Data.cwiseProduct(Matrix::Constant(mOutput->Data.rows(), mOutput->Data.cols(), 1.0) - mOutput->Data));

	Weights -= mLearningRate*mOutput->Delta*mOutput->Data.transpose();

	Vector DeltaSum(mOutput->Delta.rows());
	for (int j = 0; j < DeltaSum.size(); j++)
		DeltaSum[j] = (mOutput->Delta.row(j)).sum();
	Biases -= mLearningRate*DeltaSum;
}
