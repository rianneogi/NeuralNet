#include "Neuron.h"

Neuron::Neuron() : mInput(NULL), mOutput(NULL), mBackpropInput(NULL), mBackpropOutput(NULL), mLearningRate(1)
{
}

Neuron::Neuron(Matrix* input, Matrix* output, Matrix* bpInput, Matrix* bpOutput, Float learning_rate) : mInput(input), mOutput(output),
mBackpropInput(bpInput), mBackpropOutput(bpOutput), mLearningRate(learning_rate)
{
}

Neuron::~Neuron()
{
}
