#include "DiagNeuron.h"

DiagNeuron::DiagNeuron()
{
}

DiagNeuron::DiagNeuron(Blob* input, Blob* output, Tensor pad_value)
	: Neuron(input, output), PadValue(pad_value)
{
	assert(input->Data.mShape.size() == 4);

	BatchSize = input->Data.mShape[0];
	InputHeight = input->Data.mShape[1];
	InputWidth = input->Data.mShape[2];
	InputDepth = input->Data.mShape[3];

	assert(output->Data.mShape.size() == 2);
	OutputCols = output->Data.mShape[1];
	assert(OutputCols == InputHeight*InputDepth);
	OutputRows = output->Data.mShape[0];
	DiagCount = InputHeight*(InputWidth - 1) * 2;
	assert(OutputRows == BatchSize*DiagCount);

	assert(pad_value.mSize == InputDepth);
}

DiagNeuron::~DiagNeuron()
{
	PadValue.freemem();
}

void DiagNeuron::forward()
{
	for (uint64_t batch = 0; batch < BatchSize; batch++)
	{
		uint64_t sub_batch = 0;
		for (uint64_t y = 0; y < InputHeight; y++)
		{
			uint64_t id = 0;
			for (uint64_t t = 0; t < InputWidth; y++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), &mInput->Data(batch, y + t, y + t, 0), sizeof(Float)*InputDepth);
				id++;
			}
			sub_batch++;
		}

		for (uint64_t y = 0; y < InputHeight; y++)
		{
			uint64_t id = 0;
			for (uint64_t t = 0; t < InputWidth; y++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), &mInput->Data(batch, y - t, y + t, 0), sizeof(Float)*InputDepth);
				id++;
			}
			sub_batch++;
		}

		for (uint64_t y = 0; y < InputHeight; y++)
		{
			uint64_t id = 0;
			for (uint64_t t = 0; t < InputWidth; y++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), &mInput->Data(batch, y + t, y - t, 0), sizeof(Float)*InputDepth);
				id++;
			}
			sub_batch++;
		}

		for (uint64_t y = 0; y < InputHeight; y++)
		{
			uint64_t id = 0;
			for (uint64_t t = 0; t < InputWidth; y++)
			{
				memcpy(&mOutput->Data(batch*DiagCount + sub_batch, id), &mInput->Data(batch, y - t, y - t, 0), sizeof(Float)*InputDepth);
				id++;
			}
			sub_batch++;
		}
	}
}

void DiagNeuron::backprop()
{
}

std::vector<Blob*> DiagNeuron::getVariables()
{
	return std::vector<Blob*>();
}