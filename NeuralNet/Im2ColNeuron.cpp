#include "Im2ColNeuron.h"

Im2ColNeuron::Im2ColNeuron() : Neuron()
{
}

Im2ColNeuron::Im2ColNeuron(Blob* input, Blob* output, Float learning_rate, unsigned int kernel_width, unsigned int kernel_height) 
	: Neuron(input, output, learning_rate), KernelWidth(kernel_width), KernelHeight(kernel_height)
{
	InputWidth = input->Data.cols();
	InputHeight = input->Data.rows();

	assert(output->Data.cols() == KernelHeight*KernelWidth);
}

Im2ColNeuron::~Im2ColNeuron()
{
}

void Im2ColNeuron::forward()
{
	Matrix res(1, (InputWidth - 1 - KernelWidth)*(InputHeight - 1 - KernelHeight));
	for (unsigned int x = KernelWidth / 2; x < InputWidth - 1 - KernelWidth / 2; x++)
	{
		for (unsigned int y = KernelHeight / 2; y < InputHeight - 1 - KernelHeight / 2; y++)
		{
			Matrix m = mInput->Data.block(x, y, KernelHeight, KernelWidth).array();
			m.resize(m.cols()*m.rows(), 1);
			res << m;
		}
	}
	mOutput->Data = res;
}

void Im2ColNeuron::backprop()
{
}
