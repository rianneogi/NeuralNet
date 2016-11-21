#include "math.h"
#include <vector>
#include "conio.h"
#include "assert.h"

typedef double Float;
typedef std::vector<Float> Vector;

double sigmoid(double x)
{
	return (1.0 / (1.0 - exp(-x)));
}

class Neuron
{
public:
	Vector weights;
	double bias;

	Neuron() {}
	Neuron(const std::vector<double>& w, double b) : weights(w), bias(b) {}
	Neuron(int num_weights) : weights(num_weights), bias(0.0) {}
	~Neuron() {}

	double compute(const std::vector<double>& inputs)
	{
		assert(weights.size() == inputs.size());
		double res = bias;
		for (int i = 0; i < inputs.size(); i++)
		{
			res += weights[i] * inputs[i];
		}

		return sigmoid(res);
	}
};

class NeuralNet
{
public:
	std::vector<std::vector<Neuron>> Neurons;

	NeuralNet() {}
	~NeuralNet() {}

	void addLayer()
	{
		Neurons.push_back(std::vector<Neuron>(0));
	}

	void addNeuron(Neuron n, int layer)
	{
		Neurons[layer].push_back(n);
	}

	std::vector<double> forward(std::vector<double> inputs)
	{
		for (int i = 0; i < Neurons.size(); i++)
		{
			std::vector<double> newinputs;
			for (int j = 0; j < Neurons[i].size(); j++)
			{
				newinputs.push_back(Neurons[i][j].compute(inputs));
			}
			inputs = newinputs;
		}
		return inputs;
	}

	void backprop(Vector input, Vector output)
	{

	}
};

int main()
{
	std::vector<double> w1;
	w1.push_back(20);
	w1.push_back(20);
	Neuron n1(w1, -30);

	std::vector<double> w2;
	w2.push_back(-20);
	w2.push_back(-20);
	Neuron n2(w2, 10);

	std::vector<double> w3;
	w3.push_back(20);
	w3.push_back(20);
	Neuron n3(w3, -10);

	NeuralNet nn;
	nn.addLayer();
	nn.addNeuron(n1, 0);
	nn.addNeuron(n2, 0);
	nn.addLayer();
	nn.addNeuron(n3, 1);

	std::vector<double> inputs;
	inputs.push_back(1);
	inputs.push_back(0);

	printf("%f", nn.forward(inputs)[0]);
	_getch();

	return 0;
}