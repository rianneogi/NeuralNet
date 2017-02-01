#pragma once

#include "Neuron.h"

class FullyConnectedNeuron : public Neuron
{
public:
	unsigned int InputSize;
	unsigned int OutputSize;
	unsigned int BatchSize;
	Matrix Weights;
	Vector Biases;

	FullyConnectedNeuron(); 
	FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate);
	~FullyConnectedNeuron();

	void forward();
	void backprop();
};

