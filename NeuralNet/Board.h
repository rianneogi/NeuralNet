#pragma once

#include "ErrorFunction.h"

class Board
{
public:
	std::vector<Neuron*> Neurons;
	std::vector<Blob*> Blobs;
	ErrorFunction* ErrorFunc;
	Matrix Input;

	Board();
	~Board();

	void addNeuron(Neuron* n);
	//void addEdge(unsigned int n1, unsigned int n2);

	Matrix forward(Matrix input);
	void backprop(Matrix input, Matrix output);
};

