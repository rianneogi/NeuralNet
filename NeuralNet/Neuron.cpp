#include "Neuron.h"

Neuron::Neuron() : mInput(NULL), mOutput(NULL), mBackpropInput(NULL), mBackpropOutput(NULL)
{
}

Neuron::Neuron(Matrix* input, Matrix* output, Matrix* bpInput, Matrix* bpOutput) : mInput(input), mOutput(output),
mBackpropInput(bpInput), mBackpropOutput(bpOutput)
{
}

Neuron::~Neuron()
{
}
