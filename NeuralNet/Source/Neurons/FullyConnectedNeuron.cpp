#include "FullyConnectedNeuron.h"

FullyConnectedNeuron::FullyConnectedNeuron() : Neuron()
{
}

FullyConnectedNeuron::FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output, learning_rate)
{
	Weights = Tensor(make_shape(input->Data.cols(), output->Data.cols()));
	Biases = Tensor(make_shape(output->Data.cols()));
	for (int i = 0; i < Weights.cols(); i++)
	{
		Biases(i) = rand_init();
		for (int j = 0; j < Weights.rows(); j++)
		{
			Weights(i, j) = rand_init();
		}
	}
	InputSize = Weights.rows();
	OutputSize = Weights.cols();
	BatchSize = output->Data.rows();
	assert(input->Data.rows() == output->Data.rows());
}

FullyConnectedNeuron::~FullyConnectedNeuron()
{
}

void FullyConnectedNeuron::forward()
{
	//mOutput->Data = (Weights * mInput->Data) + (Biases.replicate(1, (mInput->Data).cols()));
	////cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Weights.rows(), mInput->Data.cols(),
	////	Weights.cols(), 1, Weights.mData, Weights.cols(), mInput->Data.mData, mInput->Data.cols(), 0, mOutput->Data.mData, mOutput->Data.cols());
	assert(mInput->Data.cols() == Weights.rows());
	assert(mOutput->Data.cols() == Weights.cols());
	assert(mOutput->Data.rows() == mInput->Data.rows());
	printf("mul\n");
	gemm(&mInput->Data, &Weights, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
	printf("done\n");
	for (unsigned int i = 0; i < mInput->Data.cols(); i++)
	{
		for (unsigned int j = 0; j < Biases.mSize; j++)
		{
			mOutput->Data(i, j) += Biases(j);
		}
	}
	printf("f\n");
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
	printf("mul1\n");
	gemm(&mOutput->Delta, &Weights, &mInput->Delta, CblasNoTrans, CblasTrans, 1, 0);
	printf("done\n");
	Tensor tmp(make_shape(Weights.rows(), Weights.cols()));
	printf("mul2\n");
	gemm(&mInput->Data, &mOutput->Delta, &tmp, CblasTrans, CblasNoTrans, mLearningRate, 0);
	printf("done\n");
	for (int i = 0; i < Weights.mSize; i++)
	{
		Weights(i) -= tmp(i);
	}

	//Biases
	Tensor tmp2(make_shape(Biases.rows(), 1));
	Tensor ones(make_shape(mOutput->Delta.cols(), 1));
	printf("mul3\n");
	gemm(&mOutput->Delta, &ones, &tmp2, CblasNoTrans, CblasNoTrans, mLearningRate, 0);
	printf("done\n");
	
	for (int i = 0; i < Biases.mSize; i++)
	{
		Biases(i) -= tmp2(i);
	}
}
