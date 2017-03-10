#pragma once
#include "..\Neuron.h"
class ReshapeNeuron :
	public Neuron
{
public:
	TensorShape InputShape;
	TensorShape OutputShape;

	ReshapeNeuron();
	ReshapeNeuron(Blob* input, Blob* output, TensorShape output_shape);
	~ReshapeNeuron();

	void forward();
	void backprop();
};

