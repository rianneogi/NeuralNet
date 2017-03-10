#pragma once
#include "../Neuron.h"
class Col2ImNeuron : public Neuron
{
public:
	Col2ImNeuron();
	Col2ImNeuron(Blob* input, Blob* output);
	~Col2ImNeuron();

	void forward();
	void backprop();
};

