#pragma once

#include "RecurrentNeuralNet.h"

//typedef Eigen::Tensor< Tensor;

class Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	Float mLearningRate;

	Neuron();
	Neuron(Blob* input, Blob* output, Float learning_rate);
	~Neuron();

	virtual void forward() = 0;
	virtual void backprop() = 0;
};

