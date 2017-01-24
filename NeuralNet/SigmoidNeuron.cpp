#include "SigmoidNeuron.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

SigmoidNeuron::SigmoidNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output, learning_rate)
{
	Weights = Matrix(output->Data.rows(), input->Data.rows());
	Biases = Vector(output->Data.rows());
	for (int i = 0; i < Weights.rows(); i++)
	{
		Biases[i] = rand_init();
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

SigmoidNeuron::~SigmoidNeuron()
{
}

void SigmoidNeuron::forward()
{
	//printf("multiplying %d %d %d %d\n", Weights.rows(), Weights.cols(), mInput->Data.rows(), mInput->Data.cols());
	//assert(mInput->Data.cols() == BatchSize && mInput->Data.rows() == InputSize);
	//assert(mOutput->Data.cols() == BatchSize && mOutput->Data.rows() == OutputSize);

	//printf("%d %d %d %d %d %d \n", InputSize, OutputSize, BatchSize, (Weights).rows(), (Weights * mInput->Data).cols(),Biases.size());

	mOutput->Data = ((Weights * mInput->Data) + (Biases.replicate(1, (mInput->Data).cols()))).unaryExpr(&sigmoid);
	//assert(mOutput->Data.cols() == BatchSize && mOutput->Data.rows() == OutputSize);
}

void SigmoidNeuron::backprop()
{
	//assert(mOutput->Delta.cols() == BatchSize && mOutput->Delta.rows() == OutputSize);
	mInput->Delta = (Weights.transpose()*mOutput->Delta).cwiseProduct(mInput->Data.cwiseProduct(Matrix::Constant(mInput->Data.rows(), mInput->Data.cols(), 1.0) - mInput->Data));
	//assert(mInput->Delta.cols() == BatchSize && mInput->Delta.rows() == InputSize);

	Weights = Weights - (mLearningRate*mOutput->Delta*mInput->Data.transpose());
	assert(Weights.cols() == InputSize && Weights.rows() == OutputSize);
	
	Vector DeltaSum(mOutput->Delta.rows());
	for (int j = 0; j < DeltaSum.size(); j++)
		DeltaSum[j] = (mOutput->Delta.row(j)).sum();
	Biases = Biases - (mLearningRate*DeltaSum);
}
