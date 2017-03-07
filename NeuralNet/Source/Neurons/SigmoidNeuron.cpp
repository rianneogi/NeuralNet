#include "SigmoidNeuron.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

SigmoidNeuron::SigmoidNeuron(Blob* input, Blob* output, Float learning_rate) : Neuron(input, output, learning_rate)
{
	assert(input->Data.cols() == output->Data.cols() && input->Data.rows() == output->Data.rows());
}

SigmoidNeuron::~SigmoidNeuron()
{
}

void SigmoidNeuron::forward()
{
	////printf("multiplying %d %d %d %d\n", Weights.rows(), Weights.cols(), mInput->Data.rows(), mInput->Data.cols());
	////assert(mInput->Data.cols() == BatchSize && mInput->Data.rows() == InputSize);
	////assert(mOutput->Data.cols() == BatchSize && mOutput->Data.rows() == OutputSize);

	////printf("%d %d %d %d %d %d \n", InputSize, OutputSize, BatchSize, (Weights).rows(), (Weights * mInput->Data).cols(),Biases.size());

	//mOutput->Data = mInput->Data.unaryExpr(&sigmoid);
	////assert(mOutput->Data.cols() == BatchSize && mOutput->Data.rows() == OutputSize);
}

void SigmoidNeuron::backprop()
{
	////assert(mOutput->Delta.cols() == BatchSize && mOutput->Delta.rows() == OutputSize);
	//mInput->Delta = (mOutput->Delta).cwiseProduct(mOutput->Data.cwiseProduct(Matrix::Constant(mOutput->Data.rows(), mOutput->Data.cols(), 1.0) - mOutput->Data));
	////assert(mInput->Delta.cols() == BatchSize && mInput->Delta.rows() == InputSize);

	////Weights = Weights - (mLearningRate*mOutput->Delta*mInput->Data.transpose());
	////assert(Weights.cols() == InputSize && Weights.rows() == OutputSize);
	//
	////Vector DeltaSum(mOutput->Delta.rows());
	////for (int j = 0; j < DeltaSum.size(); j++)
	////	DeltaSum[j] = (mOutput->Delta.row(j)).sum();

	////Biases = Biases - (mLearningRate*mOutput->Delta*Matrix::Constant(mOutput->Delta.cols(),1,1.0));
}
