#include "Im2ColNeuron.h"

Im2ColNeuron::Im2ColNeuron() : Neuron()
{
}

Im2ColNeuron::Im2ColNeuron(Blob* input, Blob* output, unsigned int field_width, unsigned int field_height)
	: Neuron(input, output), FieldWidth(field_width), FieldHeight(field_height)
{
	BatchSize = input->Data.mShape[0];

	assert(input->Data.mShape.size() == 4);

	InputDepth = input->Data.mShape[1];
	InputHeight = input->Data.mShape[2];
	InputWidth = input->Data.mShape[3];

	assert(output->Data.mShape.size() == 2);
	OutputCols = output->Data.mShape[1];
	assert(OutputCols == FieldHeight*FieldWidth*InputDepth);
	OutputRows = output->Data.mShape[0];
	assert(OutputRows == BatchSize*(InputWidth - FieldWidth + 1)*(InputHeight - FieldHeight + 1));
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
	printf("forward\n");
	assert(mInput->Data.mShape.size() == 4);
	//Works only for odd receptive fields
	for (int batch = 0; batch < BatchSize; batch++)
	{
		int id = 0;
		int sub_batch = 0;
		for (int y = FieldHeight / 2; y < InputHeight - FieldHeight / 2; y++)
		{
			for (int x = FieldWidth / 2; x < InputWidth - FieldWidth / 2; x++)
			{
				for (int d = 0; d < InputDepth; d++)
				{
					for (int i = y - FieldHeight / 2; i <= y + FieldHeight / 2; i++)
					{
						for (int j = x - FieldWidth / 2; j <= x + FieldWidth / 2; j++)
						{
							mOutput->Data(batch + sub_batch, id) = mInput->Data(batch, d, i, j);
							id++;
						}
					}
				}
				sub_batch++;
			}
		}
	}
}

void Im2ColNeuron::backprop()
{
	printf("backward\n");
}
