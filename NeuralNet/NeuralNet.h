#pragma once

#include "math.h"
#include <vector>
#include "conio.h"
#include "assert.h"
#include "time.h"

typedef double Float;
typedef std::vector<Float> Vector;

class Neuron
{
public:
	Vector weights;
	double bias;
	double delta;
	double output;
	
	Neuron();
	Neuron(const Vector & w, double b);
	Neuron(int num_weights);
	~Neuron();

	double compute(const Vector & inputs);
};

class NeuralNet
{
public:
	std::vector<std::vector<Neuron>> Neurons;
	int input_size;
	
	NeuralNet();
	NeuralNet(int i_s);
	~NeuralNet();

	void addLayer();
	void addNeuron(int layer);
	void addNeuron(Neuron n, int layer);

	Vector forward(Vector inputs);
	double backprop(Vector input, Vector output);

};

