#pragma once

#include "ErrorFunctions\MeanSquaredError.h"

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

	Tensor forward(const Tensor& input);
	Float backprop(const Tensor& input, const Tensor& output);

	Tensor predict(Tensor input);

	double train(const Tensor& inputs, const Tensor& outputs, unsigned int epochs, unsigned int batch_size);
};

