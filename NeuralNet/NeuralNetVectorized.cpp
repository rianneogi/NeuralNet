#include "NeuralNetVectorized.h"

NeuralNetVectorized::NeuralNetVectorized()
{
}

NeuralNetVectorized::NeuralNetVectorized(int input_size, double learning_rate) : InputSize(input_size), LearningRate(learning_rate)
{
}

NeuralNetVectorized::~NeuralNetVectorized()
{
}

void NeuralNetVectorized::setLayers(std::vector<unsigned int> layersizes)
{
	Weights.clear();
	for (size_t i = 0; i < layersizes.size(); i++)
	{
		unsigned int prev = InputSize;
		if (i > 0)
			prev = layersizes[i - 1];
		Weights.push_back(Matrix(prev, layersizes[i]));
	}
}

void NeuralNetVectorized::addLayer(unsigned int num_neurons)
{
	unsigned int prev = InputSize;
	if (Weights.size() > 0)
	{
		prev = Weights[Weights.size() - 1].rows();
	}
	printf("creating matrix %d %d\n", num_neurons, prev);
	Weights.push_back(Matrix(num_neurons, prev));
	Biases.push_back(Vector(num_neurons));
	Outputs.push_back(Vector(num_neurons));
	Deltas.push_back(Vector(num_neurons));
}

double sigmoid_func(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

Vector NeuralNetVectorized::forward(Vector inputs)
{
	for (size_t i = 0; i < Weights.size(); i++)
	{
		//Vector tmp = (Matrix::Constant(1) + (-inputs*LayerWeights[i]).exp()).cwiseinv();
		inputs = (Weights[i]*inputs + Biases[i]).unaryExpr(&sigmoid);
		Outputs[i] = inputs;
		//inputs = tmp;
		//printf("%d %d %d %d", inputs.rows(), inputs.cols(), LayerWeights[i].rows(), LayerWeights[i].cols());
		//inputs = LayerWeights[i]*inputs;
		//inputs.transform([](double val) { return 1.0 / (1.0 + exp(-val)); });

		/*Vector newinputs(Neurons[i].size());
		for (size_t j = 0; j < Neurons[i].size(); j++)
		{
			newinputs[j] = (Neurons[i][j].compute(inputs));
		}
		inputs = newinputs;*/
	}
	return inputs;
}

double NeuralNetVectorized::backprop(Vector input, Vector output)
{
	Vector frwd = forward(input);

	double error = 0.0;

	bool is_output = false;
	bool is_input = false;
	for (int i = Weights.size() - 1; i >= 0; i--)
	{
		if (i == Weights.size() - 1)
			is_output = true;
		else
			is_output = false;

		if (i == 0)
			is_input = true;
		else
			is_input = false;

		for (size_t j = 0; j < Weights[i].rows(); j++)
		{
			double op = Outputs[i][j];

			if (is_output) //calculate error
			{
				error += (output[j] - op)*(output[j] - op);
			}

			if (is_output)
				Deltas[i][j] = (op - output[j])*op*(1.0 - op);
			else
			{
				Deltas[i][j] = 0.0;

				for (size_t k = 0; k < Weights[i + 1].rows(); k++)
				{
					Deltas[i][j] += Weights[i + 1](k,j) * Deltas[i+1][k];
				}
				Deltas[i][j] *= op*(1.0 - op);
			}

			Biases[i][j] -= LearningRate*Deltas[i][j];
			for (size_t k = 0; k < Weights[i].cols(); k++)
			{
				if (is_input)
					Weights[i](j, k) -= LearningRate*Deltas[i][j] *input[k];
				else
					Weights[i](j, k) -= LearningRate*Deltas[i][j] *Outputs[i-1][k];
			}
		}
	}

	return error;
}

double NeuralNetVectorized::train(const std::vector<Vector>& inputs, const std::vector<Vector>& outputs, unsigned int epochs)
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

void NeuralNetVectorized::clear()
{
	Weights.clear();
	//InputSize = 0;
}

void NeuralNetVectorized::save(std::string filename) const
{
	std::fstream file(filename, std::ios::trunc | std::ios::out);
	if (file.is_open())
	{
		file << Weights.size() << " " << InputSize << "\n";
		for (size_t i = 0; i < Weights.size(); i++)
		{
			file << Weights[i].cols() << "\n";
			for (size_t j = 0; j < Weights[i].cols(); j++)
			{
				file << Biases[i][j] << " ";
				for (size_t k = 0; k < Weights[i].rows(); k++)
				{
					file << Weights[i](j,k) << " ";
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

void NeuralNetVectorized::load(std::string filename)
{
	std::fstream file(filename, std::ios::in);
	if (file.is_open())
	{
		clear();

		size_t layers, input_size;
		file >> layers >> input_size;
		//printf("layers = %d, %d", layers, input_size);
		InputSize = input_size;
		//addLayers(layers);

		for (unsigned int i = 0; i < layers; i++)
		{
			unsigned int num_neurons;
			file >> num_neurons;
			//printf("num neurons: %d\n", num_neurons);
			addLayer(num_neurons);
			for (unsigned int j = 0; j < num_neurons; j++)
			{
				//addNeuron(i);
				//printf("adding neuron to layer %d\n", i);

				//unsigned int weight_cnt = InputSize;
				//if (i > 0)
				unsigned int weight_cnt = Weights[i].cols();
				double b;
				file >> b;
				Biases[i][j] = b;
				for (unsigned int k = 0; k < weight_cnt; k++)
				{
					double w;
					file >> w;
					Weights[i](j,k) = w;
					//printf("setting weight: %f\n", w);
				}
			}
		}

		file.close();

		/*for (int i = 0; i < Weights.size(); i++)
		{
			std::cout << Weights[i] << std::endl;
		}
		for (int i = 0; i < Biases.size(); i++)
		{
			std::cout << Biases[i] << std::endl;
		}*/
	}
	else
	{
		printf("ERROR opening file: %s\n", filename.c_str());
	}
}
