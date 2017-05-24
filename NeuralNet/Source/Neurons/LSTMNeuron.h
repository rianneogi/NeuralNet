#pragma once

#include "../Neuron.h"

class LSTMNeuron : public Neuron
{
public:
	Blob* Weights_Data;
	Blob* Weights_Forget;
	Blob* Weights_Input;
	Blob* Weights_Output;

	Tensor State;
	Tensor Input;
	Tensor Output;
	Tensor Forget;

	LSTMNeuron();
	LSTMNeuron(Blob* input, Blob* output);
	~LSTMNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};