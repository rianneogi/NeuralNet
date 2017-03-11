#pragma once

#include "FullyConnectedNeuron.h"

class ConvNeuron : public Neuron
{
public:
	uint64_t InputSize;

	uint64_t OutputWidth;
	uint64_t OutputHeight;
	uint64_t OutputDepth;

	uint64_t FieldWidth;
	uint64_t FieldHeight;

	uint64_t BatchSize;

	Tensor Weights;
	Tensor Biases;

	Tensor Tmp1;
	Tensor Tmp2;
	Tensor Ones;

	Float LearningRate;

	ConvNeuron();
	ConvNeuron(Blob* input, Blob* output, Float learning_rate);
	~ConvNeuron();

	void forward();
	void backprop();
};

