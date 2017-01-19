#include "SigmoidNeuron.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

SigmoidNeuron::SigmoidNeuron(Matrix* input, Matrix* output) : Neuron(input, output)
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
}
