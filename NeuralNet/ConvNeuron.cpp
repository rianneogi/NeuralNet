#include "ConvNeuron.h"

ConvNeuron::ConvNeuron() : Neuron()
{
}

ConvNeuron::ConvNeuron(Blob* input, Blob* output, Float learning_rate, unsigned int kernel_width, unsigned int kernel_height)
	: Neuron(input, output, learning_rate), KernelWidth(kernel_width), KernelHeight(kernel_height)
{
	InputWidth = input->Data.cols();
	InputHeight = input->Data.rows();

	assert(output->Data.cols() == KernelHeight*KernelWidth);
}

ConvNeuron::~ConvNeuron()
{
}

void ConvNeuron::forward()
{
}

void ConvNeuron::backprop()
{
}