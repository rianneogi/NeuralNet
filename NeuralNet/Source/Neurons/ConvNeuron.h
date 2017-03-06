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

	unsigned int KernelWidth;
	unsigned int KernelHeight;

	unsigned int BatchSize;

	Tensor Weights;
	float Bias;

	ConvNeuron();
	ConvNeuron(Blob* input, Blob* output, Float learning_rate, unsigned int kernel_width, unsigned int kernel_height);
	~ConvNeuron();

	void forward();
	void backprop();
};

