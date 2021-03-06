#include "LeakyReLUNeuron.h"

#include <algorithm>

LeakyReLUNeuron::LeakyReLUNeuron()
{
}

LeakyReLUNeuron::LeakyReLUNeuron(Blob* input, Blob* output, Float leak_factor) 
	: mInput(input), mOutput(output), LeakFactor(leak_factor)
{
	assert(input->Data.mSize == output->Data.mSize);
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
