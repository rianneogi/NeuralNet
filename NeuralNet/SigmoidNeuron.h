#pragma once

#include "Neuron.h"

//Fully connected Sigmoid Neuron
//	Input: Matrix: Batch of input vectors
//  Output: Vector: Sigmoid nonlinearity for each of the inputs

//Forward:
//	O = 1/(1+exp(-W*I))

//Backward:
//  BO = (W^t*D)xOx(1-O)

class SigmoidNeuron : public Neuron
{
public:
	Matrix Weights;
	Vector Biases;

	SigmoidNeuron();
	SigmoidNeuron(Blob* input, Blob* output, Float learning_rate);
	~SigmoidNeuron();

	void forward();
	void backprop();
};

