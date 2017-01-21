#pragma once

#include "MeanSquaredError.h"

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
	Blob* newBlob();
	//void addEdge(unsigned int n1, unsigned int n2);

	Matrix forward(Matrix input);
	Float backprop(Matrix input, Matrix output);

	double train(const Matrix& inputs, const Matrix& outputs, unsigned int epochs, unsigned int batch_size);
};

