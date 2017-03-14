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

Blob* Board::newBlob(const TensorShape& shape)
{
	Blob* b = new Blob(shape);
	mBlobs.push_back(b);
	return b;
}

void Board::setErrorFunction(ErrorFunction* err_func)
{
	mErrorFunc = err_func;
}

Tensor Board::forward(const Tensor& input)
{
	mNeurons[0]->mInput->Data = input;
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		mNeurons[i]->forward();
	}
	return mNeurons[mNeurons.size()-1]->mOutput->Data;
}

Float Board::backprop(const Tensor& input, const Tensor& output)
{
	mNeurons[0]->mInput->Data.mData = input.mData;
	mErrorFunc->mTarget = &output;
	//printf("forward\n");

	//Forward Pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		//printf("ff: %d\n", i);
		mNeurons[i]->forward();
	}

	//Calculate Error
	double error = mErrorFunc->calculateError();

	//printf("backward\n");
	//Backward Pass
	for (int i = mNeurons.size()-1; i >= 0; i--)
	{
		//printf("bb: %d\n", i);
		mNeurons[i]->backprop();
	}

	return error;
}

Tensor Board::predict(const Tensor& input)
{
	mNeurons[0]->mInput->Data.mData = input.mData;

	//Forward Pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		mNeurons[i]->forward();
	}
	//assert(mNeurons[3]->mInput == mNeurons[3]->mOutput);
	//mNeurons[3]->backprop();
	return mNeurons[mNeurons.size()-1]->mOutput->Data;
}

double Board::train(const Tensor& inputs, const Tensor& outputs, unsigned int epochs, unsigned int batch_size)
{
	assert(inputs.rows() == outputs.rows());
	assert(inputs.rows() % batch_size == 0);
	double error = 0.0;
	printf("Started training\n");
	Clock clock;
	clock.Start();
	for (int i = 0; i < epochs; i++)
	{
		error = 0.0;
		for (int j = 0; j < inputs.rows() / batch_size; j++)
		{
			//error += backprop(inputs.block(0, batch_size*j, inputs.rows(), batch_size), outputs.block(0, batch_size*j, outputs.rows(), batch_size));
			//printf("Batch: %d\n", j);
			//printf("bs: %d\n", batch_size);
			//Tensor in = inputs.cut(batch_size*j, batch_size);
			//printf("I: %d %d\n", inputs.cut(batch_size*j, batch_size).mSize, int(in.mSelfAllocated));
			error += backprop(inputs.cut(batch_size*j, batch_size), outputs.cut(batch_size*j, batch_size));
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
