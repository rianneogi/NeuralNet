#include "Tensor.h"

Tensor::Tensor() : mData(NULL)
{
}

Tensor::Tensor(const TensorShape& shape) : mData(NULL), mShape(shape)
{
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
	printf("Allocation tensor of size: %d\n", size);
}

void Tensor::free()
{
	if (mData != NULL)
	{
		delete[] mData;
	}
}

unsigned int Tensor::rows()
{
	assert(mShape.size() >= 2);
	return mShape[1];
}

unsigned int Tensor::cols()
{
	assert(mShape.size() >= 1);
	return mShape[0];
}

TensorShape make_shape(unsigned int a)
{
	TensorShape shape;
	shape.push_back(a);
	return shape;
}

TensorShape make_shape(unsigned int a, unsigned int b)
{
	TensorShape shape;
	shape.push_back(a);
	shape.push_back(b);
	return shape;
}

TensorShape make_shape(unsigned int a, unsigned int b, unsigned int c)
{
	TensorShape shape;
	shape.push_back(a);
	shape.push_back(b);
	shape.push_back(c);
	return shape;
}

TensorShape make_shape(unsigned int a, unsigned int b, unsigned int c, unsigned int d)
{
	TensorShape shape;
	shape.push_back(a);
	shape.push_back(b);
	shape.push_back(c);
	shape.push_back(d);
	return shape;
}
