#include "Tensor.h"

Tensor::Tensor() : mData(NULL)
{
}

Tensor::Tensor(const TensorShape& shape) : mData(NULL), mShape(shape)
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	allocate();
}

Tensor::~Tensor()
{
	free();
}

Float& Tensor::operator()(unsigned int a)
{
	return mData[a];
}

Float& Tensor::operator()(unsigned int a, unsigned int b)
{
	assert(mShape.size() >= 1);
	return mData[a*mShape[0] + b];
}

Float& Tensor::operator()(unsigned int a, unsigned int b, unsigned int c)
{
	assert(mShape.size() >= 2);
	return mData[a*mShape[0]*mShape[1] + b*mShape[0] + c];
}

Float& Tensor::operator()(unsigned int a, unsigned int b, unsigned int c, unsigned int d)
{
	assert(mShape.size() >= 3);
	return mData[a*mShape[0]*mShape[1]*mShape[2] + b*mShape[0]*mShape[1] + c*mShape[0] + d];
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

void Tensor::print()
{
	for (int i = 0; i < mShape[0]; i++)
	{
		for (int j = 0; j < mShape[1]; j++)
		{
			printf("%f ", operator()(i, j));
		}
		printf("\n");
	}
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
