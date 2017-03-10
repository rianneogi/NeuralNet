#include "FullyConnectedNeuron.h"

FullyConnectedNeuron::FullyConnectedNeuron() : Neuron(), LearningRate(1)
{
}

FullyConnectedNeuron::FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output), LearningRate(learning_rate)
{
	Weights = Tensor(make_shape(input->Data.cols(), output->Data.cols()));
	Biases = Tensor(make_shape(output->Data.cols()));
	for (int i = 0; i < Weights.cols(); i++)
	{
		Biases(i) = rand_init();
		for (int j = 0; j < Weights.rows(); j++)
		{
			Weights(j, i) = rand_init();
		}
	}
	InputSize = Weights.rows();
	OutputSize = Weights.cols();
	BatchSize = output->Data.rows();
	assert(input->Data.rows() == output->Data.rows());

	Tmp1 = Tensor(make_shape(Weights.rows(), Weights.cols()));
	Tmp2 = Tensor(make_shape(Biases.rows(), 1));
	Ones = Tensor(make_shape(mOutput->Delta.cols(), 1));
	Ones.setconstant(1);
}

FullyConnectedNeuron::~FullyConnectedNeuron()
{
	Weights.freememory();
	Biases.freememory();
	Tmp1.freememory();
	Tmp2.freememory();
	Ones.freememory();
}

void FullyConnectedNeuron::forward()
{
	/*assert(mInput->Data.cols() == Weights.rows());
	assert(mOutput->Data.cols() == Weights.cols());
	assert(mOutput->Data.rows() == mInput->Data.rows());*/
	gemm(&mInput->Data, &Weights, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);;
	for (unsigned int i = 0; i < mInput->Data.rows(); i++)
	{
		for (unsigned int j = 0; j < Biases.mSize; j++)
		{
			mOutput->Data(i, j) += Biases(j);
		}
	}
}

void FullyConnectedNeuron::backprop()
{
	//Weights
	gemm(&mOutput->Delta, &Weights, &mInput->Delta, CblasNoTrans, CblasTrans, 1, 0);
	gemm(&mInput->Data, &mOutput->Delta, &Tmp1, CblasTrans, CblasNoTrans, LearningRate, 0);
	for (int i = 0; i < Weights.mSize; i++)
	{
		Weights(i) -= Tmp1(i);
	}

	//Biases
	gemm(&mOutput->Delta, &Ones, &Tmp2, CblasNoTrans, CblasNoTrans, LearningRate, 0);
	
	for (int i = 0; i < Biases.mSize; i++)
	{
		Biases(i) -= Tmp2(i);
	}
}
