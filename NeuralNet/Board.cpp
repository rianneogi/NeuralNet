#include "Board.h"

Board::Board() : ErrorFunc(nullptr)
{
}

Board::~Board()
{
	//Free memory
	delete ErrorFunc;
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		delete Neurons[i];
	}
	for (size_t i = 0; i < Blobs.size(); i++)
	{
		delete Blobs[i];
	}
}

void Board::addNeuron(Neuron* n)
{
	Neurons.push_back(n);
}

Blob* Board::newBlob()
{
	Blob* b = new Blob();
	return b;
}

void Board::setErrorFunction(ErrorFunction* err_func)
{
	ErrorFunc = err_func;
}

Matrix Board::forward(const Matrix& input)
{
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		Neurons[i]->forward();
	}
	return Neurons[Neurons.size()-1]->mOutput->Data;
}

Float Board::backprop(const Matrix& input, const Matrix& output)
{
	Neurons[0]->mInput->Data = input;
	ErrorFunc->mTarget = &output;
	//Forward Pass
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		Neurons[i]->forward();
	}

	//Calculate Error
	double error = ErrorFunc->calculateError();

	//Backward Pass
	for (int i = Neurons.size()-1; i >= 0; i--)
	{
		Neurons[i]->backprop();
	}

	return error;
}

Vector Board::predict(Vector input)
{
	Neurons[0]->mInput->Data = input;

	//Forward Pass
	for (size_t i = 0; i < Neurons.size(); i++)
	{
		Neurons[i]->forward();
	}
	return Neurons[Neurons.size()-1]->mOutput->Data;
}

double Board::train(const Matrix& inputs, const Matrix& outputs, unsigned int epochs, unsigned int batch_size)
{
	assert(inputs.cols() == outputs.cols());
	assert(inputs.cols() % batch_size == 0);
	double error = 0.0;
	printf("Started training\n");
	Clock clock;
	clock.Start();
	for (int i = 0; i < epochs; i++)
	{
		error = 0.0;
		for (int j = 0; j < inputs.cols() / batch_size; j++)
		{
			error += backprop(inputs.block(0, batch_size*j, inputs.rows(), batch_size), outputs.block(0, batch_size*j, outputs.rows(), batch_size));
		}
		/*for (int i = 0; i < inputs.size(); i++)
		{
		error += backprop(inputs[i], outputs[i]);
		}*/
		clock.Stop();
		printf("Error %d: %f, epochs per sec: %f\n", i, error, ((i + 1)*1.0) / clock.ElapsedSeconds());
		//printf("Error %d: %f\n", j, error);
	}
	printf("Done training\n");
	return error;
}
