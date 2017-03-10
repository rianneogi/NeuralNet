#pragma once

#include "FullyConnectedNeuron.h"

class ConvNeuron : public Neuron
{
public:
	unsigned int InputSize;

	unsigned int OutputWidth;
	unsigned int OutputHeight;
	unsigned int OutputDepth;

	unsigned int FieldWidth;
	unsigned int FieldHeight;

	unsigned int BatchSize;

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

