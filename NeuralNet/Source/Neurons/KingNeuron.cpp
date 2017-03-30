#include "KingNeuron.h"

KingNeuron::KingNeuron()
{
}

KingNeuron::KingNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height, Tensor pad_value)
	: Neuron(input, output), FieldWidth(field_width), FieldHeight(field_height), PadValue(pad_value)
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
	FieldCount = InputWidth*InputHeight;
	assert(OutputRows == BatchSize*FieldCount);

	assert(pad_value.mSize == InputDepth);
}

KingNeuron::~KingNeuron()
{
	PadValue.freemem();
}

void KingNeuron::forward()
{
	//works only for Field size = 3
	for (uint64_t batch = 0; batch < BatchSize; batch++)
	{
		uint64_t sub_batch = 0;
		for (uint64_t y = 0; y < InputHeight; y++)
		{
			for (uint64_t x = 0; x < InputWidth; x++)
			{
				uint64_t id = 0;
				for (uint64_t i = y - FieldHeight / 2; i <= y + FieldHeight / 2; i++)
				{
					if (i < 0 || i >= InputHeight)
					{
						memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id), PadValue.mData, InputDepth * sizeof(Float));
						memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id+InputDepth), PadValue.mData, InputDepth * sizeof(Float));
						memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id+2*InputDepth), PadValue.mData, InputDepth * sizeof(Float));
					}
					else
					{
						if (x == 0)
						{
							memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id), PadValue.mData, InputDepth * sizeof(Float));
							memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id+InputDepth), 
								&mInput->Data(batch, i, x, 0), 2*InputDepth * sizeof(Float));
						}
						else if (x == InputHeight - 1)
						{
							memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id),
								&mInput->Data(batch, i, x-1, 0), 2 * InputDepth * sizeof(Float));
							memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id+2*InputDepth), PadValue.mData, InputDepth * sizeof(Float));
						}
						else
						{
							memcpy(&mOutput->Data(batch*FieldCount + sub_batch, id),
								&mInput->Data(batch, i, x - FieldWidth / 2, 0), FieldWidth * InputDepth * sizeof(Float));
						}
					}
					id += FieldWidth*InputDepth;
				}
				sub_batch++;
			}
		}
	}
}

void KingNeuron::backprop()
{
}

std::vector<Blob*> KingNeuron::getVariables()
{
	return std::vector<Blob*>();
}