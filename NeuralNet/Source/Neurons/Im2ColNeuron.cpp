#include "Im2ColNeuron.h"

Im2ColNeuron::Im2ColNeuron() : Neuron()
{
}

Im2ColNeuron::Im2ColNeuron(Blob* input, Blob* output, Float learning_rate, unsigned int field_width, unsigned int field_height)
	: Neuron(input, output, learning_rate), FieldWidth(field_width), FieldHeight(field_height)
{
	assert(input->Data.mShape[0] == output->Data.mShape[0]);
	BatchSize = input->Data.mShape[0];

	assert(input->Data.mShape.size() == 4);

	InputDepth = input->Data.mShape[1];
	InputHeight = input->Data.mShape[2];
	InputWidth = input->Data.mShape[3];

	assert(output->Data.mShape.size() == 2);
	OutputSize = output->Data.mShape[1];
	assert(OutputSize == FieldHeight*FieldWidth*InputDepth);
}

Im2ColNeuron::~Im2ColNeuron()
{
}

void Im2ColNeuron::forward()
{
	/*Matrix res(1, (InputWidth - 1 - KernelWidth)*(InputHeight - 1 - KernelHeight));
	for (unsigned int x = KernelWidth / 2; x < InputWidth - 1 - KernelWidth / 2; x++)
	{
		for (unsigned int y = KernelHeight / 2; y < InputHeight - 1 - KernelHeight / 2; y++)
		{
			Matrix m = mInput->Data.block(x, y, KernelHeight, KernelWidth).array();
			m.resize(m.cols()*m.rows(), 1);
			res << m;
		}
	}
	mOutput->Data = res;*/
	for (int batch = 0; batch < BatchSize; batch++)
	{
		int id = 0;
		for (int d = 0; d < InputDepth; d++)
		{
			for (int y = FieldHeight / 2; y < InputHeight - 1 - FieldHeight / 2; y++)
			{
				for (int x = FieldWidth / 2; x < InputWidth - 1 - FieldWidth / 2; x++)
				{
					mOutput->Data(batch, id) = mInput->Data(batch, d, y, x);
					id++;
				}
			}
		}
	}
}

void Im2ColNeuron::backprop()
{
}
