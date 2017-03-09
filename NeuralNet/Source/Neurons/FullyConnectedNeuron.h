#pragma once

#include "../Neuron.h"

class FullyConnectedNeuron : public Neuron
{
public:
	unsigned int InputSize;
	unsigned int OutputSize;
	unsigned int BatchSize;
	Tensor Weights;
	Tensor Biases;
	Tensor Tmp1;
	Tensor Tmp2;
	Tensor Ones;

	FullyConnectedNeuron(); 
	FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate);
	~FullyConnectedNeuron();

	void forward();
	void backprop();
};

