#include "Im2ColNeuron.h"

Im2ColNeuron::Im2ColNeuron() : Neuron()
{
}

Im2ColNeuron::Im2ColNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height)
	: Neuron(input, output), FieldWidth(field_width), FieldHeight(field_height)
{
	BatchSize = input->Data.mShape[0];

	assert(input->Data.mShape.size() == 4);

	InputHeight = input->Data.mShape[1];
	InputWidth = input->Data.mShape[2];
	InputDepth = input->Data.mShape[3];

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
	//Works only for odd sized receptive fields
	for (uint64_t batch = 0; batch < BatchSize; batch++)
	{
		uint64_t sub_batch = 0;
		for (uint64_t y = FieldHeight / 2; y < InputHeight - FieldHeight / 2; y++)
		{
			for (uint64_t x = FieldWidth / 2; x < InputWidth - FieldWidth / 2; x++)
			{
				uint64_t id = 0;
				for (uint64_t i = y - FieldHeight / 2; i <= y + FieldHeight / 2; i++)
				{
					memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id),
						&mInput->Data(batch, i, x - FieldWidth / 2, 0), FieldWidth * InputDepth * sizeof(Float));
					//uint64_t xid = id;
					id += FieldWidth*InputDepth;
					/*for (uint64_t j = x - FieldWidth / 2; j <= x + FieldWidth / 2; j++)
					{
					assert(mOutput->Data(batch*FieldCount + sub_batch, xid) == mInput->Data(batch, d, i, j));
					xid++;
					}*/
				}
				sub_batch++;
			}
		}
	}
	//mInput->Data.print_raw();
}

void Im2ColNeuron::backprop()
{
}

std::vector<Blob*> Im2ColNeuron::getVariables()
{
	return std::vector<Blob*>();
}
