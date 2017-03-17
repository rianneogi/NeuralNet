#include "FullyConnectedNeuron.h"

FullyConnectedNeuron::FullyConnectedNeuron() : Neuron(), LearningRate(1)
{
}

FullyConnectedNeuron::FullyConnectedNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output), LearningRate(learning_rate)
{
	Weights = new Blob(make_shape(input->Data.cols(), output->Data.cols()));
	Biases = new Blob(make_shape(1, output->Data.cols()));
	for (int i = 0; i < Weights->Data.cols(); i++)
	{
		Biases->Data(i) = rand_init();
		for (int j = 0; j < Weights->Data.rows(); j++)
		{
			Weights->Data(j, i) = rand_init();
		}
	}
	InputSize = Weights->Data.rows();
	OutputSize = Weights->Data.cols();
	BatchSize = output->Data.rows();
	assert(input->Data.rows() == output->Data.rows());

	//WeightsDelta = Tensor(make_shape(Weights->Data.rows(), Weights->Data.cols()));
	//BiasesDelta = Tensor(make_shape(1, Biases->Data.mSize));
	Ones = Tensor(make_shape(1, BatchSize));
	Ones.setconstant(1);
}

FullyConnectedNeuron::~FullyConnectedNeuron()
{
	/*Weights.freememory();
	Biases.freememory();
	WeightsDelta.freememory();
	BiasesDelta.freememory();*/
	delete Weights;
	delete Biases;
	Ones.freememory();
}

void FullyConnectedNeuron::forward()
{
#ifdef USE_GPU
	gemm(&mInput->Data, &Weights->Data, &mOutput->Data, clblasNoTrans, clblasNoTrans, 1, 0);
#else
	gemm_cpu(&mInput->Data, &Weights->Data, &mOutput->Data, CblasNoTrans, CblasNoTrans, 1, 0);
#endif
	for (unsigned int i = 0; i < mInput->Data.rows(); i++)
	{
		for (unsigned int j = 0; j < Biases->Data.mSize; j++)
		{
			mOutput->Data(i, j) += Biases->Data(j);
		}
	}
}

void FullyConnectedNeuron::backprop()
{
#ifdef USE_GPU
	//Weights
	gemm(&mOutput->Delta, &Weights->Data, &mInput->Delta, clblasNoTrans, clblasTrans, 1, 0);
	gemm(&mInput->Data, &mOutput->Delta, &Weights->Delta, clblasTrans, clblasNoTrans, 1, 0);

	//Biases
	gemm(&Ones, &mOutput->Delta, &Biases->Delta, clblasNoTrans, clblasNoTrans, 1, 0);
#else
	//Weights
	gemm_cpu(&mOutput->Delta, &Weights->Data, &mInput->Delta, CblasNoTrans, CblasTrans, 1, 0);
	gemm_cpu(&mInput->Data, &mOutput->Delta, &Weights->Delta, CblasTrans, CblasNoTrans, 1, 0);

	//Biases
	gemm_cpu(&Ones, &mOutput->Delta, &Biases->Delta, CblasNoTrans, CblasNoTrans, 1, 0);
#endif
}

std::vector<Blob*> FullyConnectedNeuron::getVariables()
{
	std::vector<Blob*> v;
	v.push_back(Weights);
	v.push_back(Biases);
	return v;
}
