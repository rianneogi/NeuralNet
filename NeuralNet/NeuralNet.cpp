#include "NeuralNet.h"

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
	for (size_t i = 0; i < inputs.size(); i++)
	{
		res += weights[i] * inputs[i];
	}
	//printf("old output: %f\n", output);
	output = sigmoid(res);
	//printf("new output: %f\n", output);
	return output;
}

NeuralNet::NeuralNet() {}

NeuralNet::NeuralNet(int input_size, double learning_rate) : InputSize(input_size), LearningRate(learning_rate) {}

NeuralNet::~NeuralNet() {}

void NeuralNet::addLayer()
{
	Neurons.push_back(std::vector<Neuron>(0));
}

void NeuralNet::addLayers(unsigned int count)
{
	for (unsigned int i = 0; i < count; i++)
	{
		addLayer();
	}
}

void NeuralNet::addNeuron(unsigned int layer)
{
	if (layer == 0)
	{
		Neuron n(InputSize);
		Neurons[layer].push_back(n);
	}
	else
	{
		Neuron n(Neurons[layer - 1].size());
		Neurons[layer].push_back(n);
	}
}

void NeuralNet::addNeuron(Neuron n, unsigned int layer)
{
	Neurons[layer].push_back(n);
}

Vector NeuralNet::forward(Vector inputs)
{
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		Vector newinputs;
		for (size_t j = 0; j < Neurons[i].size(); j++)
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

	double error = 0.0;

	bool is_output = false;
	bool is_input = true;
	for (int i = Neurons.size() - 1; i >= 0; i--)
	{
		if (i == Neurons.size() - 1)
			is_output = true;
		else
			is_output = false;

		if (i == 0)
			is_input = true;
		else
			is_input = false;

		for (size_t j = 0; j < Neurons[i].size(); j++)
		{
			double op = Neurons[i][j].output;

			if (is_output) //calculate error
			{
				error += (output[j] - op)*(output[j] - op);
			}

			if(is_output)
				Neurons[i][j].delta = (op - output[j])*op*(1.0 - op);
			else
			{
				Neurons[i][j].delta = 0.0;

				for (size_t k = 0; k < Neurons[i + 1].size(); k++)
				{
					Neurons[i][j].delta += Neurons[i+1][k].weights[j]*Neurons[i+1][k].delta;
				}
				Neurons[i][j].delta *= op*(1.0 - op);
			}
			
			Neurons[i][j].bias -= LearningRate*Neurons[i][j].delta;
			for (size_t k = 0; k < Neurons[i][j].weights.size(); k++)
			{
				if(is_input)
					Neurons[i][j].weights[k] -= LearningRate*Neurons[i][j].delta*input[k];
				else
					Neurons[i][j].weights[k] -= LearningRate*Neurons[i][j].delta*Neurons[i-1][k].output;
			}
		}
	}

	return error;

	//double error = 0.0;
	//for (size_t i = 0; i < output.size(); i++)
	//{
	//	double x = Neurons[Neurons.size() - 1][i].output;
	//	error += (output[i] - x)*(output[i] - x);
	//	Neurons[Neurons.size() - 1][i].delta = (output[i] - x)*x*(1.0-x);
	//	//printf("Desired output %d: %f, current output: %f\n", i, output[i], x);
	//	//printf("For neuron %d %d, setting delta %f\n", Neurons.size() - 1, i, Neurons[Neurons.size() - 1][i].delta);
	//}
	////printf("\nError: %f\n\n", error);

	//for (size_t i = 0; i < Neurons.size() - 1; i++)
	//{
	//	for (size_t j = 0; j < Neurons[i].size(); j++)
	//	{
	//		double delta = 0.0;
	//		for (size_t k = 0; k < Neurons[i + 1].size(); k++)
	//		{
	//			delta += Neurons[i + 1][k].weights[j] * Neurons[i + 1][k].delta;
	//			//delta += Neurons[i + 1][k].bias * Neurons[i + 1][k].delta;
	//		}

	//		double x = Neurons[i][j].output;
	//		Neurons[i][j].delta = (delta)*x*(1.0-x);
	//		//printf("For neuron %d %d, setting delta %f %f\n", i, j, Neurons[i][j].delta, x);
	//	}
	//}
	////printf("\n");

	//for (size_t i = 0; i < Neurons.size(); i++)
	//{
	//	for (size_t j = 0; j < Neurons[i].size(); j++)
	//	{
	//		//printf("Updating weight for Neuron: %d %d\n", i, j);
	//		//printf("old bias: %f\n", Neurons[i][j].bias);
	//		Neurons[i][j].bias += LearningRate*Neurons[i][j].delta;
	//		//printf("new bias: %f\n", Neurons[i][j].bias);
	//		for (size_t k = 0; k < Neurons[i][j].weights.size(); k++)
	//		{
	//			double val = input[k];
	//			if (i > 0)
	//			{
	//				val = Neurons[i - 1][k].output;
	//			}
	//			//printf("input: %f, delta: %f\n", val, Neurons[i][j].delta);
	//			//printf("old weight: %f\n", Neurons[i][j].weights[k]);
	//			Neurons[i][j].weights[k] += LearningRate*val*Neurons[i][j].delta;
	//			//printf("new weight: %f\n", Neurons[i][j].weights[k]);
	//		}
	//		//printf("\n");
	//	}
	//}
	//return error;
}

double NeuralNet::train(const std::vector<Vector>& inputs, const std::vector<Vector>& outputs, unsigned int epochs)
{
	assert(inputs.size() == outputs.size());
	double error = 0.0;
	printf("Started training\n");
	Clock clock;
	clock.Start();
	for (int j = 0; j < epochs; j++)
	{
		error = 0.0;
		for (int i = 0; i < inputs.size(); i++)
		{
			error += backprop(inputs[i], outputs[i]);
		}
		clock.Stop();
		printf("Error %d: %f, epochs per sec: %f\n", j, error, (j*1.0) / clock.ElapsedSeconds());
		//printf("Error %d: %f\n", j, error);
	}
	printf("Done training\n");
	return error;
}

void NeuralNet::clear()
{
	Neurons.clear();
	InputSize = 0;
}

void NeuralNet::save(std::string filename) const
{
	std::fstream file(filename, std::ios::trunc | std::ios::out);
	if (file.is_open())
	{
		file << Neurons.size() << " " << InputSize << "\n";
		for (size_t i = 0; i < Neurons.size(); i++)
		{
			file << Neurons[i].size() << "\n";
			for (size_t j = 0; j < Neurons[i].size(); j++)
			{
				for (size_t k = 0; k < Neurons[i][j].weights.size(); k++)
				{
					file << Neurons[i][j].weights[k] << " ";
				}
				file << "\n";
			}
		}
		file.close();
	}
	else
	{
		printf("ERROR opening file: %s\n", filename.c_str());
	}
}

void NeuralNet::load(std::string filename)
{
	std::fstream file(filename, std::ios::in);
	if (file.is_open())
	{
		clear();

		size_t layers, input_size;
		file >> layers >> input_size;
		//printf("layers = %d, %d", layers, input_size);
		InputSize = input_size;
		addLayers(layers);

		for (unsigned int i = 0; i < layers; i++)
		{
			unsigned int num_neurons;
			file >> num_neurons;
			//printf("num neurons: %d\n", num_neurons);

			for (unsigned int j = 0; j < num_neurons; j++)
			{
				addNeuron(i);
				//printf("adding neuron to layer %d\n", i);

				unsigned int weight_cnt = InputSize;
				if (i > 0)
					weight_cnt = Neurons[i - 1].size();

				for (unsigned int k = 0; k < weight_cnt; k++)
				{
					double w;
					file >> w;
					Neurons[i][j].weights[k] = w;
					//printf("setting weight: %f\n", w);
				}
			}
		}

		file.close();
	}
	else
	{
		printf("ERROR opening file: %s\n", filename.c_str());
	}
}
