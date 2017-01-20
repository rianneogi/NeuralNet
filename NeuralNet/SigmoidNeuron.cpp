#include "SigmoidNeuron.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

SigmoidNeuron::SigmoidNeuron(Matrix* input, Matrix* output, Matrix* bpInput, Matrix* bpOutput, Float learning_rate) 
	: Neuron(input, output, bpInput, bpOutput, learning_rate)
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
	*mOutput = ((Weights * *mInput) + (Biases.replicate(1, (*mInput).cols()))).unaryExpr(&sigmoid);
}

void SigmoidNeuron::backprop()
{
	*mBackpropOutput = (Weights.transpose()*(*mBackpropInput)).cwiseProduct(mOutput->cwiseProduct(Matrix::Constant(mOutput->rows(), mOutput->cols(), 1.0) - *mOutput));

	Weights -= mLearningRate*(*mBackpropInput)*mOutput->transpose();

	Vector DeltaSum(mBackpropInput->rows());
	for (int j = 0; j < DeltaSum.size(); j++)
		DeltaSum[j] = (mBackpropInput->row(j)).sum();
	Biases -= mLearningRate*DeltaSum;
}
