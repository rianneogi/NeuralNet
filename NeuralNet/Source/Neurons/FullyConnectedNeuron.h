#pragma once

#include "../Neuron.h"

class FullyConnectedNeuron : public Neuron
{
public:
	uint64_t InputSize;
	uint64_t OutputSize;
	uint64_t BatchSize;
	Blob* Weights;
	Blob* Biases;
	/*Tensor WeightsDelta;
	Tensor BiasesDelta;*/
	Tensor Ones;

	Float LearningRate;

	FullyConnectedNeuron(); 
	FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate);
	~FullyConnectedNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};

