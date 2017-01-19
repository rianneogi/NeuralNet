#include "Neuron.h"

Neuron::Neuron() : mInput(NULL), mOutput(NULL)
{
}

Neuron::Neuron(Matrix * input, Matrix * output) : mInput(input), mOutput(output)
{
}

Neuron::~Neuron()
{
}
