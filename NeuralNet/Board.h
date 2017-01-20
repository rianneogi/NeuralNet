#pragma once

#include "Neuron.h"

class Board
{
public:
	std::vector<Neuron*> Neurons;
	Matrix Inputs;
	Matrix Deltas;
	Matrix Outputs;

	Board();
	~Board();

	void addNeuron(Neuron* n);
	//void addEdge(unsigned int n1, unsigned int n2);

	Matrix forward(Matrix input);
	void backprop(Matrix input, Matrix output);
};

