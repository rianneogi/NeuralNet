#pragma once

#include "../Neuron.h"

class StepNeuron : public Neuron
{
public:
	StepNeuron();
	StepNeuron(Blob* input, Blob* output);
	~StepNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};
