#pragma once

#include "NeuralNetVectorized.h"

class RecurrentNeuralNet
{
public:
	Matrix WeightsHH;
	Matrix WeightsIH;
	Matrix WeightsHO;

	double LearningRate;

	/*unsigned int InputSize;
	unsigned int OutputSize;
	unsigned int HiddenSize;*/

	RecurrentNeuralNet();
	RecurrentNeuralNet(unsigned int input_size, unsigned int output_size, unsigned int hidden_size, double learning_rate);
	~RecurrentNeuralNet();

	Vector forward(Vector input);
	double backprop(Vector input, Vector output);
	double train(Matrix inputs, Matrix outputs);

	void load(std::string filename);
	void save(std::string filename) const;
};

