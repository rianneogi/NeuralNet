#include "ConvNeuron.h"

ConvNeuron::ConvNeuron() : Neuron()
{
}

ConvNeuron::ConvNeuron(Blob* input, Blob* output, Float learning_rate)
	: Neuron(input, output, learning_rate)
{
	assert(input->Data.mShape[0] == output->Data.mShape[0]);
	BatchSize = input->Data.mShape[0];

	InputSize = input->Data.mShape[1];

	OutputDepth = output->Data.mShape[1];
	//OutputHeight = output->Data.mShape[2];
	//OutputWidth = output->Data.mShape[3];

	Weights = Tensor(make_shape(InputSize, OutputDepth));
	Biases = Tensor(make_shape(OutputDepth));

	Tmp1 = Tensor(make_shape(Weights.rows(), Weights.cols()));
	Tmp2 = Tensor(make_shape(Biases.rows(), 1));
	Ones = Tensor(make_shape(mOutput->Delta.cols(), 1));
	Ones.setconstant(1);

	//assert(output->Data.cols() == FieldHeight*FieldWidth);
}

ConvNeuron::~ConvNeuron()
{
}

void ConvNeuron::forward()
{
	gemm(&mInput->Data, &Weights, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
	for (unsigned int i = 0; i < mInput->Data.mShape[0]; i++)
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
	gemm(&mInput->Data, &mOutput->Delta, &Tmp1, CblasTrans, CblasNoTrans, mLearningRate, 0);
	for (int i = 0; i < Weights.mSize; i++)
	{
		Weights(i) -= Tmp1(i);
	}

	//Biases
	gemm(&mOutput->Delta, &Ones, &Tmp2, CblasNoTrans, CblasNoTrans, mLearningRate, 0);

	for (int i = 0; i < Biases.mSize; i++)
	{
		Biases(i) -= Tmp2(i);
	}
}