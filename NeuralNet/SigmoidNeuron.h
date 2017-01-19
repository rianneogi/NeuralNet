#pragma once

#include "Neuron.h"

class SigmoidNeuron : public Neuron
{
public:
	Matrix Weights;
	Vector Biases;

	SigmoidNeuron();
	SigmoidNeuron(Matrix* input, Matrix* output);
	~SigmoidNeuron();

	void forward();
	void backprop();
};

