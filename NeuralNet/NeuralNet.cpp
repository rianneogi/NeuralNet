#include "NeuralNet.h"

double clamp(double x)
{
	return x > 0.99 ? 0.99 : (x < 0.01 ? 0.01 : x);
}

double sigmoid(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

Neuron::Neuron() {}

Neuron::Neuron(const Vector& w, double b) : weights(w), bias(b) {}

Neuron::Neuron(int num_weights) : weights(num_weights), bias(0.5)
{
	for (int i = 0; i < num_weights; i++)
	{
		//weights[i] = 0.0;
		weights[i] = (((rand() % 1024) - 512)*1.0) / 1024.0;
	}
}
Neuron::~Neuron() {}

double Neuron::compute(const Vector& inputs)
{
	assert(weights.size() == inputs.size());
	double res = bias;
	for (int i = 0; i < inputs.size(); i++)
	{
		res += weights[i] * inputs[i];
	}
	//printf("old output: %f\n", output);
	output = sigmoid(res);
	//printf("new output: %f\n", output);
	return output;
}

NeuralNet::NeuralNet() {}

NeuralNet::NeuralNet(int i_s) : input_size(i_s) {}

NeuralNet::~NeuralNet() {}

void NeuralNet::addLayer()
{
	Neurons.push_back(std::vector<Neuron>(0));
}

void NeuralNet::addNeuron(int layer)
{
	if (layer == 0)
	{
		Neuron n(input_size);
		Neurons[layer].push_back(n);
	}
	else
	{
		Neuron n(Neurons[layer - 1].size());
		Neurons[layer].push_back(n);
	}
}

void NeuralNet::addNeuron(Neuron n, int layer)
{
	Neurons[layer].push_back(n);
}

Vector NeuralNet::forward(Vector inputs)
{
	for (int i = 0; i < Neurons.size(); i++)
	{
		Vector newinputs;
		for (int j = 0; j < Neurons[i].size(); j++)
		{
			newinputs.push_back(Neurons[i][j].compute(inputs));
		}
		inputs = newinputs;
	}
	return inputs;
}

double NeuralNet::backprop(Vector input, Vector output)
{
	Vector frwd = forward(input);
	//printf("forward: %f\n", frwd[0]);

	double error = 0.0;
	for (int i = 0; i < output.size(); i++)
	{
		double x = Neurons[Neurons.size() - 1][i].output;
		error += (output[i] - x)*(output[i] - x);
		Neurons[Neurons.size() - 1][i].delta = ((output[i] - x) * x * (1.0 - x));
		//printf("Desired output %d: %f, current output: %f\n", i, output[i], x);
		//printf("For neuron %d %d, setting delta %f\n", Neurons.size() - 1, i, Neurons[Neurons.size() - 1][i].delta);
	}
	//printf("\nError: %f\n\n", error);
	for (int i = 0; i < Neurons.size() - 1; i++)
	{
		for (int j = 0; j < Neurons[i].size(); j++)
		{
			double delta = 0.0;
			for (int k = 0; k < Neurons[i + 1].size(); k++)
			{
				delta += Neurons[i + 1][k].weights[j] * Neurons[i + 1][k].delta;
				//delta += Neurons[i + 1][k].bias * Neurons[i + 1][k].delta;
			}

			double x = Neurons[i][j].output;
			Neurons[i][j].delta = (delta * x * (1.0 - x));
			//printf("For neuron %d %d, setting delta %f %f\n", i, j, Neurons[i][j].delta, x);
		}
	}
	//printf("\n");
	double alpha = 1;
	for (int i = 0; i < Neurons.size(); i++)
	{
		for (int j = 0; j < Neurons[i].size(); j++)
		{
			//printf("Updating weight for Neuron: %d %d\n", i, j);
			//printf("old bias: %f\n", Neurons[i][j].bias);
			Neurons[i][j].bias += alpha*Neurons[i][j].delta;
			//printf("new bias: %f\n", Neurons[i][j].bias);
			for (int k = 0; k < Neurons[i][j].weights.size(); k++)
			{
				double val = input[k];
				if (i > 0)
				{
					val = Neurons[i - 1][k].output;
				}
				//printf("input: %f, delta: %f\n", val, Neurons[i][j].delta);
				//printf("old weight: %f\n", Neurons[i][j].weights[k]);
				Neurons[i][j].weights[k] += alpha*val*Neurons[i][j].delta;
				//printf("new weight: %f\n", Neurons[i][j].weights[k]);
			}
			//printf("\n");
		}
	}
	return error;
}