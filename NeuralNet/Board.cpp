#include "Board.h"

Board::Board()
{
}

Board::~Board()
{
}

void Board::addNeuron(Neuron* n)
{
	Neurons.push_back(n);
}

void Board::addEdge(unsigned int n1, unsigned int n2)
{
}

Matrix Board::forward(Matrix input)
{
	Matrix tmp = input;
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		tmp = Neurons[i]->compute(tmp);
	}
	return tmp;
}

void Board::backprop(Matrix input, Matrix output)
{
}
