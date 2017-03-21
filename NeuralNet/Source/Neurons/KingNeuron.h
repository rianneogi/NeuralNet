#pragma once
#include "../Neuron.h"

class KingNeuron : public Neuron
{
public:
	uint64_t InputWidth;
	uint64_t InputHeight;
	uint64_t InputDepth;

	uint64_t OutputCols;
	uint64_t OutputRows;

	uint64_t FieldWidth;
	uint64_t FieldHeight;

	uint64_t FieldCount;

	uint64_t BatchSize;

	Tensor PadValue;

	KingNeuron();
	KingNeuron(Blob* input, Blob* output, uint64_t field_width, uint64_t field_height, Tensor pad_value);
	~KingNeuron();

	void forward();
	void backprop();
	std::vector<Blob*> getVariables();
};