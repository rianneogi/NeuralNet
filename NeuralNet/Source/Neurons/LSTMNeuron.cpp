#include "LSTMNeuron.h"

LSTMNeuron::LSTMNeuron()
{
}

LSTMNeuron::LSTMNeuron(Blob* input, Blob* output) : Neuron(input, output)
{
}

LSTMNeuron::~LSTMNeuron()
{
}

void LSTMNeuron::forward()
{
}

void LSTMNeuron::backprop()
{
}

std::vector<Blob*> LSTMNeuron::getVariables()
{
	return std::vector<Blob*>();
}
