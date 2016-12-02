#pragma once

#include "NeuralNet.h"

class NeuralNetVectorized
{
public:
	//std::vector<std::vector<Neuron>> Neurons;
	unsigned int InputSize;
	double LearningRate;
	unsigned int BatchSize;

	std::vector<Matrix> Weights;
	std::vector<Vector> Biases;
	std::vector<Matrix> Outputs;
	std::vector<Matrix> Deltas;

	NeuralNetVectorized();
	NeuralNetVectorized(unsigned int input_size, double learning_rate);
	~NeuralNetVectorized();

	void setLayers(std::vector<unsigned int> layersizes);
	void addLayer(unsigned int num_neurons);

	Vector forward(Vector inputs);
	Matrix forward(Matrix inputs);
	double backprop(Vector input, Vector output);
	double backprop(Matrix inputs, Matrix outputs);
	double train(const Matrix& inputs, const Matrix& outputs, unsigned int epochs);

	void clear();

	void save(std::string filename) const;
	void load(std::string filename);
};

