#pragma once

#include "NeuralNetVectorized.h"

class RecurrentNeuralNet
{
public:
	Matrix WeightsHH;
	Matrix WeightsIH;
	Matrix WeightsHO;

	Vector BiasesH;
	Vector BiasesO;

	std::vector<Matrix> OutputsH;
	std::vector<Matrix> OutputsO;
	std::vector<Matrix> Deltas;

	double LearningRate;

	/*unsigned int InputSize;
	unsigned int OutputSize;
	unsigned int HiddenSize;*/

	RecurrentNeuralNet();
	RecurrentNeuralNet(unsigned int input_size, unsigned int output_size, unsigned int hidden_size, double learning_rate);
	~RecurrentNeuralNet();

	void forward(std::vector<Matrix> input, unsigned int time_steps);
	double backprop(std::vector<Matrix> input, std::vector<Matrix> output, unsigned int time_steps);
	double train(std::vector<Matrix> inputs, std::vector<Matrix> outputs, unsigned int time_steps);

	void load(std::string filename);
	void save(std::string filename) const;
};