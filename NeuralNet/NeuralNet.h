#pragma once

#include "LogisticRegression.h"

class NeuronNN
{
public:
	Vector weights;
	double bias;
	double delta;
	double output;
	
	NeuronNN();
	NeuronNN(const Vector& w, double b);
	NeuronNN(int num_weights);
	~NeuronNN();

	double compute(const Vector& inputs);
};

class NeuralNet
{
public:
	std::vector<std::vector<NeuronNN>> Neurons;
	int InputSize;
	double LearningRate;
	
	NeuralNet();
	NeuralNet(int input_size, double learning_rate);
	~NeuralNet();

	void addLayer();
	void addLayers(unsigned int count);
	void addNeuron(unsigned int layer);
	void addNeuron(NeuronNN n, unsigned int layer);

	Vector forward(Vector inputs);
	double backprop(Vector input, Vector output);
	double train(const Matrix& inputs, const Matrix& outputs, unsigned int epochs);

	void clear();

	void save(std::string filename) const;
	void load(std::string filename);
};

