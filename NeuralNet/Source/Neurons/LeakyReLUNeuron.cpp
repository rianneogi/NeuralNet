#include "LeakyReLUNeuron.h"

#include <algorithm>

LeakyReLUNeuron::LeakyReLUNeuron()
{
}

LeakyReLUNeuron::LeakyReLUNeuron(Blob* input, Blob* output, Float leak_factor) 
	: Neuron(input, output), LeakFactor(leak_factor)
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

LeakyReLUNeuron::~LeakyReLUNeuron()
{
}

void LeakyReLUNeuron::forward()
{
	for (uint64_t i = 0; i < mInput->Data.mSize; i++)
	{
		mOutput->Data(i) = std::max(LeakFactor*mInput->Data(i), mInput->Data(i));
	}
}

void LeakyReLUNeuron::backprop()
{
	for (uint64_t i = 0; i < mInput->Data.mSize; i++)
	{
		mInput->Delta(i) += mOutput->Data(i) < 0.0? LeakFactor*mOutput->Delta(i): mOutput->Delta(i);
	}
}

std::vector<Blob*> LeakyReLUNeuron::getVariables()
{
	return std::vector<Blob*>();
}
