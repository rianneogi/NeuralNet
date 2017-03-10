#include "ReshapeNeuron.h"

ReshapeNeuron::ReshapeNeuron()
{
}

ReshapeNeuron::ReshapeNeuron(Blob* input, Blob* output) : Neuron(input, output)
{
	assert(input->Data.mSize == output->Data.mSize);
}

ReshapeNeuron::~ReshapeNeuron()
{
}

void ReshapeNeuron::forward()
{
	memcpy(&mOutput->Data, &mInput->Data, sizeof(Float)*mInput->Data.mSize);
}

void ReshapeNeuron::backprop()
{
	memcpy(&mInput->Delta, &mOutput->Delta, sizeof(Float)*mInput->Delta.mSize);
}
