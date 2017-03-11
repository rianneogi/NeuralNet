#pragma once

#include "../Neuron.h"

class FullyConnectedNeuron : public Neuron
{
public:
	uint64_t InputSize;
	uint64_t OutputSize;
	uint64_t BatchSize;
	Tensor Weights;
	Tensor Biases;
	Tensor Tmp1;
	Tensor Tmp2;
	Tensor Ones;

	Float LearningRate;

	FullyConnectedNeuron(); 
	FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate);
	~FullyConnectedNeuron();

	void forward();
	void backprop();
};

