#include "FullyConnectedNeuron.h"

FullyConnectedNeuron::FullyConnectedNeuron() : Neuron()
{
}

FullyConnectedNeuron::FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output, learning_rate)
{
	Weights = Tensor(make_shape(input->Data.rows(), output->Data.rows()));
	Biases = Tensor(make_shape(output->Data.rows()));
	for (int i = 0; i < Weights.rows(); i++)
	{
		Biases(i) = rand_init();
		for (int j = 0; j < Weights.cols(); j++)
		{
			Weights(i, j) = rand_init();
		}
	}
	InputSize = Weights.cols();
	OutputSize = Weights.rows();
	BatchSize = output->Data.cols();
	assert(input->Data.cols() == output->Data.cols());
}

FullyConnectedNeuron::~FullyConnectedNeuron()
{
}

void FullyConnectedNeuron::forward()
{
	//mOutput->Data = (Weights * mInput->Data) + (Biases.replicate(1, (mInput->Data).cols()));
	////cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Weights.rows(), mInput->Data.cols(),
	////	Weights.cols(), 1, Weights.mData, Weights.cols(), mInput->Data.mData, mInput->Data.cols(), 0, mOutput->Data.mData, mOutput->Data.cols());
	gemm(&mInput->Data, &Weights, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
	for (unsigned int i = 0; i < mInput->Data.cols(); i++)
	{
		for (unsigned int j = 0; j < Biases.mSize; j++)
		{
			mOutput->Data(i, j) += Biases(j);
		}
	}
}

void FullyConnectedNeuron::backprop()
{
	////assert(mOutput->Delta.cols() == BatchSize && mOutput->Delta.rows() == OutputSize);
	//mInput->Delta = Weights.transpose()*mOutput->Delta;
	////assert(mInput->Delta.cols() == BatchSize && mInput->Delta.rows() == InputSize);

	//Weights = Weights - (mLearningRate*mOutput->Delta*mInput->Data.transpose());
	//assert(Weights.cols() == InputSize && Weights.rows() == OutputSize);

	////Vector DeltaSum(mOutput->Delta.rows());
	////for (int j = 0; j < DeltaSum.size(); j++)
	////	DeltaSum[j] = (mOutput->Delta.row(j)).sum();

	//Biases = Biases - (mLearningRate*mOutput->Delta*Matrix::Constant(mOutput->Delta.cols(), 1, 1.0));

	//Weights
	gemm(&Weights, &mOutput->Delta, &mInput->Delta, CblasTrans, CblasNoTrans, 1, 0);

	Tensor tmp(make_shape(Weights.cols(), Weights.rows()));
	gemm(&mOutput->Delta, &mInput->Data, &tmp, CblasNoTrans, CblasTrans, mLearningRate, 0);

	for (int i = 0; i < Weights.mSize; i++)
	{
		Weights(i) -= tmp(i);
	}

	//Biases
	Tensor tmp2(make_shape(Biases.cols(), Biases.rows()));
	Tensor ones(make_shape(mOutput->Delta.cols(), 1));
	gemm(&mOutput->Delta, &ones, &tmp2, CblasNoTrans, CblasNoTrans, mLearningRate, 0);
	
	for (int i = 0; i < Biases.mSize; i++)
	{
		Biases(i) -= tmp2(i);
	}
}
