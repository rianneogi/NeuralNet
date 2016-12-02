#pragma once

#include "LogisticRegression.h"

class Neuron
{
public:
	Vector weights;
	double bias;
	double delta;
	double output;
	
	Neuron();
	Neuron(const Vector& w, double b);
	Neuron(int num_weights);
	~Neuron();

	double compute(const Vector& inputs);
};

class NeuralNet
{
public:
	std::vector<std::vector<Neuron>> Neurons;
	int InputSize;
	double LearningRate;
	
	NeuralNet();
	NeuralNet(int input_size, double learning_rate);
	~NeuralNet();

	void addLayer();
	void addLayers(unsigned int count);
	void addNeuron(unsigned int layer);
	void addNeuron(Neuron n, unsigned int layer);

	Vector forward(Vector inputs);
	double backprop(Vector input, Vector output);
	double train(const Matrix& inputs, const Matrix& outputs, unsigned int epochs);

	void clear();

	void save(std::string filename) const;
	void load(std::string filename);
};

