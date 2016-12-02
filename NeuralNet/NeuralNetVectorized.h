#pragma once

#include "NeuralNet.h"

class NeuralNetVectorized
{
public:
	//std::vector<std::vector<Neuron>> Neurons;
	unsigned int InputSize;
	double LearningRate;

	std::vector<Matrix> Weights;
	std::vector<Vector> Biases;
	std::vector<Vector> Outputs;
	std::vector<Vector> Deltas;

	NeuralNetVectorized();
	NeuralNetVectorized(int input_size, double learning_rate);
	~NeuralNetVectorized();

	void setLayers(std::vector<unsigned int> layersizes);
	void addLayer(unsigned num_neurons);
	//void addNeuron(unsigned int layer);
	//void addNeuron(Neuron n, unsigned int layer);

	Vector forward(Vector inputs);
	double backprop(Vector input, Vector output);
	double train(const std::vector<Vector>& inputs, const std::vector<Vector>& outputs, unsigned int epochs);

	void clear();

	void save(std::string filename) const;
	void load(std::string filename);
};

