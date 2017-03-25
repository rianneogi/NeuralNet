#pragma once

#include "Optimizer.h"

class Board
{
public:
	std::vector<Neuron*> mNeurons;
	std::vector<Blob*> mBlobs;
	std::vector<ErrorFunction*> mErrorFuncs;
	Optimizer* mOptimizer;
	//Matrix Input;

	Board();
	~Board();

	void addNeuron(Neuron* n);
	Blob* newBlob(const TensorShape& shape);
	void addErrorFunction(ErrorFunction* err_func);
	void setOptimizer(Optimizer* optimizer);
	//void addEdge(unsigned int n1, unsigned int n2);

	Tensor forward(const Tensor& input);
	Float backprop(const Tensor& input, const Tensor& output);
	Float backprop(const Tensor& input, const std::vector<Tensor*>& output);

	Tensor predict(const Tensor& input);

	double train(const Tensor& inputs, const Tensor& outputs, unsigned int epochs, unsigned int batch_size);

	void save_variables(std::string filename);
	void load_variables(std::string filename);
};

