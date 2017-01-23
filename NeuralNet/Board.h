#pragma once

#include "MeanSquaredError.h"

class Board
{
public:
	std::vector<Neuron*> mNeurons;
	std::vector<Blob*> mBlobs;
	ErrorFunction* mErrorFunc;
	//Matrix Input;

	Board();
	~Board();

	void addNeuron(Neuron* n);
	Blob* newBlob(unsigned int rows, unsigned int cols);
	void setErrorFunction(ErrorFunction* err_func);
	//void addEdge(unsigned int n1, unsigned int n2);

	Matrix forward(const Matrix& input);
	Float backprop(const Matrix& input, const Matrix& output);

	Vector predict(Vector input);

	double train(const Matrix& inputs, const Matrix& outputs, unsigned int epochs, unsigned int batch_size);
};

