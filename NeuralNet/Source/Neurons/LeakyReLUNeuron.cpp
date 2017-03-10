#include "LeakyReLUNeuron.h"

#include <algorithm>

LeakyReLUNeuron::LeakyReLUNeuron()
{
}

LeakyReLUNeuron::LeakyReLUNeuron(Blob* input, Blob* output, Float leak_factor) 
	: Neuron(input, output), LeakFactor(leak_factor)
{
}

LeakyReLUNeuron::~LeakyReLUNeuron()
{
}

void LeakyReLUNeuron::forward()
{
	for (int i = 0; i < mInput->Data.mSize; i++)
	{
		mOutput->Data(i) = std::max(LeakFactor*mInput->Data(i), mInput->Data(i));
	}
}

void LeakyReLUNeuron::backprop()
{
	for (int i = 0; i < mInput->Data.mSize; i++)
	{
		mInput->Delta(i) = mOutput->Data(i) < 0.0? LeakFactor*mOutput->Delta(i): 1.0*mOutput->Delta(i);
	}
}
