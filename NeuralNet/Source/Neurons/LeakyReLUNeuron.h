#pragma once
#include "..\Neuron.h"
class LeakyReLUNeuron :
	public Neuron
{
public:
	Float LeakFactor;
	
	LeakyReLUNeuron();
	LeakyReLUNeuron(Blob* input, Blob* output, Float leak_factor = 0);
	~LeakyReLUNeuron();

	void forward();
	void backprop();
};
