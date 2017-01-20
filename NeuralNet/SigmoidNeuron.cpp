#include "SigmoidNeuron.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

SigmoidNeuron::SigmoidNeuron(Matrix* input, Matrix* output, Matrix* bpInput, Matrix* bpOutput) : Neuron(input, output, bpInput, bpOutput)
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
}
