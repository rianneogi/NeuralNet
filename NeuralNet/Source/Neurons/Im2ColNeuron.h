#pragma once
#include "../Neuron.h"
class Im2ColNeuron : public Neuron
{
public:
	unsigned int InputWidth;
	unsigned int InputHeight;

	unsigned int KernelWidth;
	unsigned int KernelHeight;
	//unsigned int KernelDepth;
	//unsigned int KernelStride;

	Im2ColNeuron();
	Im2ColNeuron(Blob* input, Blob* output, Float learning_rate, unsigned int kernel_width, unsigned int kernel_height);
	~Im2ColNeuron();

	void forward();
	void backprop();
};

