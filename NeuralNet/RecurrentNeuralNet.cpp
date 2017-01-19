#include "RecurrentNeuralNet.h"

RecurrentNeuralNet::RecurrentNeuralNet()
{
}

RecurrentNeuralNet::RecurrentNeuralNet(unsigned int input_size, unsigned int output_size, unsigned int hidden_size, double learning_rate)
	: LearningRate(learning_rate), WeightsHH(hidden_size,hidden_size), WeightsIH(hidden_size, input_size), WeightsHO(output_size, hidden_size),
	BiasesH(hidden_size), BiasesO(output_size)
{
}

RecurrentNeuralNet::~RecurrentNeuralNet()
{
}

void RecurrentNeuralNet::forward(std::vector<Matrix> input, unsigned int time_steps)
{
	for (unsigned int i = 0; i < time_steps; i++)
	{
		if (i == 0)
		{
			OutputsH[i] = ((WeightsIH*input[i]) + (BiasesH.replicate(1, input[i].cols()))).unaryExpr(&sigmoid);
		}
		else
		{
			OutputsH[i] = ((WeightsHH * OutputsH[i - 1]) + (WeightsIH*input[i]) + (BiasesH.replicate(1, input[i].cols()))).unaryExpr(&sigmoid);
		}

		OutputsO[i] = WeightsHO*OutputsH[i] + BiasesO.replicate(1, OutputsH[i].cols());
	}
}

double RecurrentNeuralNet::backprop(std::vector<Matrix> input, std::vector<Matrix> output, unsigned int time_steps)
{
	double error = 0.0;
	for (unsigned int i = time_steps - 1; i >= 0; i--)
	{
		error += ((OutputsO[i] - output[i]).cwiseProduct(OutputsO[i] - output[i])).sum();

		Matrix delta_o = (OutputsO[i] - output[i]).cwiseProduct(OutputsO[i].cwiseProduct(
			Matrix::Constant(OutputsO[i].rows(), OutputsO[i].cols(), 1.0) - OutputsO[i]));

		WeightsHO -= LearningRate*delta_o*OutputsH[i];
		BiasesH -= LearningRate*delta_o;

		Deltas[i] = (delta_o * WeightsHO).cwiseProduct(
			OutputsH[i].cwiseProduct(Matrix::Constant(OutputsH[i].rows(), OutputsH[i].cols(), 1.0) - OutputsH[i]));

		if (i != 0)
		{
			WeightsHH -= LearningRate*Deltas[i] * OutputsH[i - 1];
		}
		WeightsIH -= LearningRate*Deltas[i] * input[i];
		BiasesH -= LearningRate*Deltas[i];
	}
	return error;
}

void RecurrentNeuralNet::load(std::string filename)
{
	std::fstream file(filename, std::ios::in);
	if (file.is_open())
	{
		unsigned int input_size, hidden_size, output_size;
		file >> input_size >> hidden_size >> output_size;
		for (unsigned int i = 0; i < hidden_size; i++)
		{
			for (unsigned int j = 0; j < input_size; j++)
			{
				double w;
				file >> w;
				WeightsIH(i, j) = w;
			}
		}
		for (unsigned int i = 0; i < hidden_size; i++)
		{
			for (unsigned int j = 0; j < hidden_size; j++)
			{
				double w;
				file >> w;
				WeightsHH(i, j) = w;
			}
		}
		for (unsigned int i = 0; i < output_size; i++)
		{
			for (unsigned int j = 0; j < hidden_size; j++)
			{
				double w;
				file >> w;
				WeightsHO(i, j) = w;
			}
		}
		file.close();
	}
	else
	{
		printf("cant open file: %s", filename.c_str());
	}
}

void RecurrentNeuralNet::save(std::string filename) const
{
	std::fstream file(filename, std::ios::trunc | std::ios::out);
	if (file.is_open())
	{
		file << WeightsIH.cols() << " " << WeightsHH.rows() << " " << WeightsHO.rows() << "\n";
		for (unsigned int i = 0; i < WeightsIH.rows(); i++)
		{
			for (unsigned int j = 0; j < WeightsIH.cols(); j++)
			{
				file << WeightsIH(i, j) << " ";
			}
			file << "\n";
		}
		for (unsigned int i = 0; i < WeightsHH.rows(); i++)
		{
			for (unsigned int j = 0; j < WeightsHH.cols(); j++)
			{
				file << WeightsHH(i, j) << " ";
			}
			file << "\n";
		}
		for (unsigned int i = 0; i < WeightsHO.rows(); i++)
		{
			for (unsigned int j = 0; j < WeightsHO.cols(); j++)
			{
				file << WeightsHO(i, j) << " ";
			}
			file << "\n";
		}
		file.close();
	}
	else
	{
		printf("cant open file: %s", filename.c_str());
	}
}
