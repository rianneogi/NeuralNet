#include "Neuron.h"

Neuron::Neuron() : mInput(NULL), mOutput(NULL), mLearningRate(1)
{
}

Neuron::Neuron(Blob* input, Blob* output, Float learning_rate) : mInput(input), mOutput(output), mLearningRate(learning_rate)
{
}

Neuron::~Neuron()
{
}
