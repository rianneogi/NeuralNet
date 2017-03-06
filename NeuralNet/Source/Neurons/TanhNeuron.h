#pragma once

#include "ConvNeuron.h"

//Fully connected Tanh Neuron
//	Input: Matrix: Batch of input vectors
//  Output: Vector: Tanh nonlinearity for each of the inputs

//Forward:
//	O = tanh(W*I)

//Backward:
//  BO = (W^t*D)x(1-O^2)

class TanhNeuron : public Neuron
{
public:
	TanhNeuron();
	TanhNeuron(Blob* input, Blob* output, Float learning_rate);
	~TanhNeuron();

	void forward();
	void backprop();
};

