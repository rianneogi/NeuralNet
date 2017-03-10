#pragma once
#include "..\Neuron.h"
class ReshapeNeuron :
	public Neuron
{
public:
	ReshapeNeuron();
	ReshapeNeuron(Blob* input, Blob* output);
	~ReshapeNeuron();
};

