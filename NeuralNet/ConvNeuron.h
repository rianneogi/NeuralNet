#pragma once

#include "FullyConnectedNeuron.h"

class ConvNeuron : public Neuron
{
public:
	unsigned int InputWidth;
	unsigned int InputHeight;
	unsigned int InputDepth;

	unsigned int OutputWidth;
	unsigned int OutputHeight;
	unsigned int OutputDepth;

	unsigned int ConvWidth;
	unsigned int ConvHeight;

	unsigned int BatchSize;

	Matrix Weights;
	float Bias;

	ConvNeuron();
	ConvNeuron(Blob* input, Blob* output, Float learning_rate);
	~ConvNeuron();

	void forward();
	void backprop();
};

