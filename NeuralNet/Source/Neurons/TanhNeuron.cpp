#include "TanhNeuron.h"

TanhNeuron::TanhNeuron() : Neuron()
{
}

TanhNeuron::TanhNeuron(Blob* input, Blob* output) : Neuron(input, output)
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

TanhNeuron::~TanhNeuron()
{
}

void TanhNeuron::forward()
{
	for (uint64_t i = 0; i < mOutput->Data.mSize; i++)
	{
		mOutput->Data(i) = tanh(mInput->Data(i));
	}
}

void TanhNeuron::backprop()
{
	for (uint64_t i = 0; i < mInput->Delta.mSize; i++)
	{
		mInput->Delta(i) += mOutput->Delta(i)*(1.0 - mOutput->Data(i)*mOutput->Data(i));
	}
}

std::vector<Blob*> TanhNeuron::getVariables()
{
	return std::vector<Blob*>();
}
