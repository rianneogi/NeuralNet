#include "Im2ColNeuron.h"

Im2ColNeuron::Im2ColNeuron() : Neuron()
{
}

Im2ColNeuron::Im2ColNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height)
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
	FieldCount = (InputWidth - FieldWidth + 1)*(InputHeight - FieldHeight + 1);
	assert(OutputRows == BatchSize*FieldCount);
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
	assert(mInput->Data.mShape.size() == 4);
	//Works only for odd receptive fields
	for (uint64_t batch = 0; batch < BatchSize; batch++)
	{
		int sub_batch = 0;
		for (uint64_t y = FieldHeight / 2; y < InputHeight - FieldHeight / 2; y++)
		{
			int id = 0;
			for (uint64_t x = FieldWidth / 2; x < InputWidth - FieldWidth / 2; x++)
			{
				for (uint64_t d = 0; d < InputDepth; d++)
				{
					for (uint64_t i = y - FieldHeight / 2; i <= y + FieldHeight / 2; i++)
					{
						for (uint64_t j = x - FieldWidth / 2; j <= x + FieldWidth / 2; j++)
						{
							mOutput->Data(batch*FieldCount + sub_batch, id) = mInput->Data(batch, d, i, j);
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
}
