#include "ConvNeuron.h"

ConvNeuron::ConvNeuron() : Neuron(), LearningRate(1)
{
}

ConvNeuron::ConvNeuron(Blob* input, Blob* output, Float learning_rate)
	: Neuron(input, output), LearningRate(learning_rate)
{
	//assert(input->Data.mShape[0] == output->Data.mShape[0]);
	BatchSize = input->Data.mShape[0];

	InputSize = input->Data.mShape[1];

	OutputDepth = output->Data.mShape[1];
	//OutputHeight = output->Data.mShape[2];
	//OutputWidth = output->Data.mShape[3];

	Weights = Tensor(make_shape(InputSize, OutputDepth));
	Biases = Tensor(make_shape(OutputDepth));
	for (int i = 0; i < Weights.cols(); i++)
	{
		Biases(i) = rand_init();
		for (int j = 0; j < Weights.rows(); j++)
		{
			Weights(j, i) = rand_init();
		}
	}

	Tmp1 = Tensor(make_shape(Weights.rows(), Weights.cols()));
	Tmp2 = Tensor(make_shape(1, Biases.mSize));
	Ones = Tensor(make_shape(1, BatchSize));
	Ones.setconstant(1);

	//assert(output->Data.cols() == FieldHeight*FieldWidth);
}

ConvNeuron::~ConvNeuron()
{
	Weights.freememory();
	Biases.freememory();
	Tmp1.freememory();
	Tmp2.freememory();
	Ones.freememory();
}

void ConvNeuron::forward()
{
	gemm(&mInput->Data, &Weights, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
	for (unsigned int i = 0; i < mOutput->Data.mShape[0]; i++)
	{
		for (unsigned int j = 0; j < Biases.mSize; j++)
		{
			mOutput->Data(i, j) += Biases(j);
		}
	}
}

void ConvNeuron::backprop()
{
	gemm(&mOutput->Delta, &Weights, &mInput->Delta, CblasNoTrans, CblasTrans, 1, 0);
	gemm(&mInput->Data, &mOutput->Delta, &Tmp1, CblasTrans, CblasNoTrans, LearningRate, 0);
	for (int i = 0; i < Weights.mSize; i++)
	{
		Weights(i) -= Tmp1(i);
	}

	//Biases
	gemm(&Ones, &mOutput->Delta, &Tmp2, CblasNoTrans, CblasNoTrans, LearningRate, 0);

	for (int i = 0; i < Biases.mSize; i++)
	{
		Biases(i) -= Tmp2(i);
	}
}