#include "Neuron.h"

Neuron::Neuron() : mInput(NULL), mOutput(NULL)
{
	printf("WARNING: default constructor for neuron called\n");
}

Neuron::Neuron(Blob* input, Blob* output) : mInput(input), mOutput(output)
{
}

Neuron::~Neuron()
{
}
