#include "Neuron.h"

Neuron::Neuron() : mInput(NULL), mOutput(NULL), mLearningRate(1)
{
	printf("WARNING: default constructor for neuron called\n");
}

Neuron::Neuron(Blob* input, Blob* output, Float learning_rate) : mInput(input), mOutput(output), mLearningRate(learning_rate)
{
}

Neuron::~Neuron()
{
}
