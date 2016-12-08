#include "RecurrentNeuralNet.h"

RecurrentNeuralNet::RecurrentNeuralNet()
{
}

RecurrentNeuralNet::RecurrentNeuralNet(unsigned int input_size, unsigned int output_size, unsigned int hidden_size, double learning_rate)
	: LearningRate(learning_rate), WeightsHH(hidden_size,hidden_size), WeightsIH(hidden_size, input_size), WeightsHO(output_size, hidden_size)
{
}

RecurrentNeuralNet::~RecurrentNeuralNet()
{
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
