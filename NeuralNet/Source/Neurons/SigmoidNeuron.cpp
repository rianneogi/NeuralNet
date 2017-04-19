#include "SigmoidNeuron.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

SigmoidNeuron::SigmoidNeuron(Blob* input, Blob* output) : Neuron(input, output)
{
	assert(input->Data.mSize == output->Data.mSize);

	if (input->Data.mLD != input->Data.mShape[input->Data.mShape.size() - 1])
	{
		printf("WARNING: Input data size doesnt match LD\n");
	}
	if (output->Data.mLD != output->Data.mShape[output->Data.mShape.size() - 1])
	{
		printf("WARNING: Output data size doesnt match LD\n");
	}
}

SigmoidNeuron::~SigmoidNeuron()
{
}

void SigmoidNeuron::forward()
{
	for (uint64_t i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = sigmoid(mInput->Data(i));
	}
}

void SigmoidNeuron::backprop()
{
	for (uint64_t i = 0; i < mInput->Delta.mSize; i++)
	{
		mInput->Delta(i) += mOutput->Delta(i)*mOutput->Data(i)*(1.0 - mOutput->Data(i));
	}
}

std::vector<Blob*> SigmoidNeuron::getVariables()
{
	return std::vector<Blob*>();
}
