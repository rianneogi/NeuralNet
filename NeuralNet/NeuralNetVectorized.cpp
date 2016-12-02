#include "NeuralNetVectorized.h"

NeuralNetVectorized::NeuralNetVectorized()
{
}

NeuralNetVectorized::NeuralNetVectorized(unsigned int input_size, double learning_rate) : InputSize(input_size), LearningRate(learning_rate)
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
	Outputs.push_back(Matrix(num_neurons, BatchSize));
	Deltas.push_back(Matrix(num_neurons, BatchSize));
}

double sigmoid_func(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

Vector NeuralNetVectorized::predict(Vector inputs)
{
	for (size_t i = 0; i < Weights.size(); i++)
	{
		inputs = (Weights[i]*inputs + Biases[i]).unaryExpr(&sigmoid);
		Outputs[i] = inputs;
	}
	return inputs;
}

Matrix NeuralNetVectorized::forward(Matrix inputs)
{
	assert(inputs.cols() == BatchSize);
	for (size_t i = 0; i < Weights.size(); i++)
	{
		if (i == 0)
		{
			//printf("%d %d %d %d\n", Weights[i].rows(), Weights[i].cols(), inputs.rows(),inputs.cols());
			Outputs[i] = ((Weights[i] * inputs) + (Biases[i].replicate(1, inputs.cols()))).unaryExpr(&sigmoid);
		}
		else
		{
			Outputs[i] = ((Weights[i] * Outputs[i-1]) + (Biases[i].replicate(1, Outputs[i-1].cols()))).unaryExpr(&sigmoid);
		}
	}
	return inputs;
}

double NeuralNetVectorized::backprop(Vector input, Vector output)
{
	//Vector frwd = forward(input);

	//double error = 0.0;

	//bool is_output = false;
	//bool is_input = false;
	//for (int i = Weights.size() - 1; i >= 0; i--)
	//{
	//	if (i == Weights.size() - 1)
	//		is_output = true;
	//	else
	//		is_output = false;

	//	if (i == 0)
	//		is_input = true;
	//	else
	//		is_input = false;

	//	for (size_t j = 0; j < Weights[i].rows(); j++)
	//	{
	//		double op = Outputs[i][j];

	//		if (is_output) //calculate error
	//		{
	//			error += (output[j] - op)*(output[j] - op);
	//		}

	//		if (is_output)
	//			Deltas[i][j] = (op - output[j])*op*(1.0 - op);
	//		else
	//		{
	//			Deltas[i][j] = 0.0;

	//			for (size_t k = 0; k < Weights[i + 1].rows(); k++)
	//			{
	//				Deltas[i][j] += Weights[i + 1](k,j) * Deltas[i+1][k];
	//			}
	//			Deltas[i][j] *= op*(1.0 - op);
	//		}

	//		Biases[i][j] -= LearningRate*Deltas[i][j];
	//		for (size_t k = 0; k < Weights[i].cols(); k++)
	//		{
	//			if (is_input)
	//				Weights[i](j, k) -= LearningRate*Deltas[i][j] *input[k];
	//			else
	//				Weights[i](j, k) -= LearningRate*Deltas[i][j] *Outputs[i-1][k];
	//		}
	//	}
	//}

	//return error;
	return 0;
}

double NeuralNetVectorized::backprop(Matrix inputs, Matrix outputs)
{
	Matrix frwd = forward(inputs);

	double error = ((outputs - Outputs[Weights.size()-1]).cwiseProduct(outputs - Outputs[Weights.size()-1])).sum();

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
		
		if (is_output)
		{
			Deltas[i] = (Outputs[i] - outputs).cwiseProduct(Outputs[i].cwiseProduct(Matrix::Constant(Outputs[i].rows(), Outputs[i].cols(), 1.0) - Outputs[i]));
		}
		else
		{
			//TODO: optimize transposes
			//printf("%d %d %d %d\n", Deltas[i + 1].rows(), Deltas[i + 1].cols(), Weights[i+1].rows(), Weights[i+1].cols());
			Deltas[i] = ((Deltas[i+1].transpose()*Weights[i + 1]).transpose()).cwiseProduct(Outputs[i].cwiseProduct(Matrix::Constant(Outputs[i].rows(), Outputs[i].cols(), 1.0) - Outputs[i]));
		}

		Vector DeltaSum(Deltas[i].rows());
		for (int j = 0; j < DeltaSum.size(); j++)
			DeltaSum[j] = (Deltas[i].row(j)).sum();
		Biases[i] -= LearningRate*DeltaSum;
		if (is_input)
		{
			//printf("%d %d %d %d\n", Deltas[i + 1].rows(), Deltas[i + 1].cols(), Weights[i + 1].rows(), Weights[i + 1].cols());
			Weights[i] -= LearningRate*Deltas[i]*inputs.transpose();
		}
		else
		{
			//printf("%d %d %d %d\n", Outputs[i - 1].rows(), Outputs[i - 1].cols(), Deltas[i].rows(), Deltas[i].cols());
			Weights[i] -= LearningRate*Deltas[i] * Outputs[i - 1].transpose();
		}

		//for (size_t j = 0; j < Weights[i].rows(); j++)
		//{
		//	double op = Outputs[i][j];

		//	//if (is_output) //calculate error
		//	//{
		//	//	error += (output[j] - op)*(output[j] - op);
		//	//}

		//	/*if (is_output)
		//		Deltas[i][j] = (op - output[j])*op*(1.0 - op);
		//	else
		//	{
		//		Deltas[i][j] = Weights[i + 1] * Deltas[i + 1];
		//		Deltas[i][j] = 0.0;

		//		for (size_t k = 0; k < Weights[i + 1].rows(); k++)
		//		{
		//			Deltas[i][j] += Weights[i + 1](k, j) * Deltas[i + 1][k];
		//		}
		//		Deltas[i][j] *= op*(1.0 - op);
		//	}*/

		//	Biases[i][j] -= LearningRate*Deltas[i][j];
		//	for (size_t k = 0; k < Weights[i].cols(); k++)
		//	{
		//		if (is_input)
		//			Weights[i](j, k) -= LearningRate*Deltas[i][j] * input[k];
		//		else
		//			Weights[i](j, k) -= LearningRate*Deltas[i][j] * Outputs[i - 1][k];
		//	}
		//}
	}

	return error;
}

double NeuralNetVectorized::train(const Matrix& inputs, const Matrix& outputs, unsigned int epochs)
{
	assert(inputs.cols() == outputs.cols());
	double error = 0.0;
	printf("Started training\n");
	Clock clock;
	clock.Start();
	assert(BatchSize == inputs.cols());
	for (int j = 0; j < epochs; j++)
	{
		error = backprop(inputs, outputs);
		/*for (int i = 0; i < inputs.size(); i++)
		{
			error += backprop(inputs[i], outputs[i]);
		}*/
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
