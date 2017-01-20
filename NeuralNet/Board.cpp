#include "Board.h"

Board::Board()
{
}

Board::~Board()
{
	//Free memory
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		delete Neurons[i];
	}
}

void Board::addNeuron(Neuron* n)
{
	Neurons.push_back(n);
}

Matrix Board::forward(Matrix input)
{
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		Neurons[i]->forward();
	}
	return Neurons[Neurons.size()-1]->mOutput->Data;
}

void Board::backprop(Matrix input, Matrix output)
{
	//Forward Pass
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		Neurons[i]->forward();
	}

	//Backward Pass
	for (int i = Neurons.size()-1; i >= 0; i--)
	{
		Neurons[i]->backward();
	}
}
