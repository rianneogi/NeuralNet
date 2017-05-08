#pragma once

#include "Neuron.h"

class Initializer
{
public:
	Initializer();
	~Initializer();

	virtual Float get_value(uint64_t n) = 0;
};
