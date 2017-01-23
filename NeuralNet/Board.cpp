#include "Board.h"

Board::Board() : mErrorFunc(nullptr)
{
}

Board::~Board()
{
	//Free memory
	delete mErrorFunc;
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		delete mNeurons[i];
	}
	for (size_t i = 0; i < mBlobs.size(); i++)
	{
		delete mBlobs[i];
	}
}

void Board::addNeuron(Neuron* n)
{
	mNeurons.push_back(n);
}

Blob* Board::newBlob(unsigned int rows, unsigned int cols)
{
	Blob* b = new Blob(rows, cols);
	return b;
}

void Board::setErrorFunction(ErrorFunction* err_func)
{
	mErrorFunc = err_func;
}

Matrix Board::forward(const Matrix& input)
{
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		mNeurons[i]->forward();
	}
	return mNeurons[mNeurons.size()-1]->mOutput->Data;
}

Float Board::backprop(const Matrix& input, const Matrix& output)
{
	mNeurons[0]->mInput->Data = input;
	mErrorFunc->mTarget = &output;
	//Forward Pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		mNeurons[i]->forward();
	}

	//Calculate Error
	double error = mErrorFunc->calculateError();

	//Backward Pass
	for (int i = mNeurons.size()-1; i >= 0; i--)
	{
		mNeurons[i]->backprop();
	}

	return error;
}

Vector Board::predict(Vector input)
{
	mNeurons[0]->mInput->Data = input;

	//Forward Pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		mNeurons[i]->forward();
	}
	return mNeurons[mNeurons.size()-1]->mOutput->Data;
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
