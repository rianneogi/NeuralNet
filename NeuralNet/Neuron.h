#pragma once

#include "RecurrentNeuralNet.h"

//typedef Eigen::Tensor< Tensor;

class Neuron
{
public:
	Matrix* mInput;
	Matrix* mOutput;

	Neuron();
	Neuron(Matrix* input, Matrix* output);
	~Neuron();

	virtual void forward() = 0;
	virtual void backward() = 0;
};

