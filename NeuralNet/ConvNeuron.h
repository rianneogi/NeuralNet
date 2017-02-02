#pragma once

#include "FullyConnectedNeuron.h"

class ConvNeuron : public Neuron
{
public:
	ConvNeuron();
	ConvNeuron(Blob* input, Blob* output, Float learning_rate);
	~ConvNeuron();

	void forward();
	void backprop();
};

