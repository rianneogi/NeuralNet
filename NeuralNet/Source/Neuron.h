#pragma once

#include "Blob.h"

class Neuron
{
public:
	Blob* mInput;
	Blob* mOutput;

	Neuron();
	Neuron(Blob* input, Blob* output);
	~Neuron();

	virtual void forward() = 0;
	virtual void backprop() = 0;
	virtual std::vector<Blob*> getVariables() = 0;
};

