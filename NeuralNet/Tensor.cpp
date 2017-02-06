#include "Tensor.h"

Tensor::Tensor() : mData(NULL)
{
}

Tensor::Tensor(unsigned int a) : mData(NULL)
{
	mShape.push_back(a);
	allocate();
}

Tensor::Tensor(unsigned int a, unsigned int b) : mData(NULL)
{
	mShape.push_back(a);
	mShape.push_back(b);
	allocate();
}

Tensor::Tensor(unsigned int a, unsigned int b, unsigned int c) : mData(NULL)
{
	mShape.push_back(a);
	mShape.push_back(b);
	mShape.push_back(c);
	allocate();
}

Tensor::Tensor(unsigned int a, unsigned int b, unsigned int c, unsigned int d) : mData(NULL)
{
	mShape.push_back(a);
	mShape.push_back(b);
	mShape.push_back(c);
	mShape.push_back(d);
	allocate();
}

Tensor::~Tensor()
{
	free();
}

void Tensor::allocate()
{
	unsigned int size = 1;
	for (unsigned int x : mShape)
	{
		size *= x;
	}
	mData = new Float[size];
}

void Tensor::free()
{
	if (mData != NULL)
	{
		delete[] mData;
	}
}
