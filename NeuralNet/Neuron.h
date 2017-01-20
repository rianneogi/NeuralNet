#pragma once

#include "RecurrentNeuralNet.h"

//typedef Eigen::Tensor< Tensor;

class Neuron
{
public:
	Matrix* mInput;
	Matrix* mOutput;
	Matrix* mBackpropInput;
	Matrix* mBackpropOutput;

	Float mLearningRate;

	Neuron();
	Neuron(Matrix* input, Matrix* output, Matrix* bpInput, Matrix* bpOutput, Float learning_rate);
	~Neuron();

	virtual void forward() = 0;
	virtual void backward() = 0;
};

