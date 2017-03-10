#pragma once

#include "../Neuron.h"

//Converts a 4D Tensor into a 2D Tensor for matrix multiplication
//Input: BatchSize x InputDepth x InputHeight x InputWidth
//Output: BatchSize*(InputWidth - FieldWidth + 1)*(InputHeight - FieldHeight + 1) x FieldWidth*FieldHeight*InputDepth

class Im2ColNeuron : public Neuron
{
public:
	unsigned int InputWidth;
	unsigned int InputHeight;
	unsigned int InputDepth;

	unsigned int OutputCols;
	unsigned int OutputRows;

	unsigned int FieldWidth;
	unsigned int FieldHeight;

	unsigned int BatchSize;

	Im2ColNeuron();
	Im2ColNeuron(Blob* input, Blob* output, unsigned int field_width, unsigned int field_height);
	~Im2ColNeuron();

	void forward();
	void backprop();
};

