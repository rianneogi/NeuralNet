#include "Tensor.h"

#define USE_MALLOC

Tensor::Tensor() : mData(NULL), mSize(0)
{
}

Tensor::Tensor(const TensorShape& shape) : mData(NULL), mShape(shape), mSize(1)
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	unsigned int mSize = 1;
	for (unsigned int x : mShape)
	{
		mSize *= x;
	}
	allocate();
}

Tensor::~Tensor()
{
	freememory();
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
#ifdef USE_MALLOC
	mData = (Float*)malloc(mSize * sizeof(Float));
	if (mData == NULL)
	{
		printf("ERROR: Cant allocate memory for tensor, Size: %d\n", mSize);
	}
#else
	mData = new Float[mSize];
#endif
	
	printf("Allocation tensor of size: %d\n", mSize);
}

void Tensor::freememory()
{
	if (mData != NULL)
	{
#ifdef USE_MALLOC
		free(mData);
		mData = NULL;
#else
		delete[] mData;
#endif
	}
}

void Tensor::setzero()
{
	memset(mData, 0, sizeof(Float)*mSize);
}

void Tensor::setidentity()
{
	setzero();
	assert(mShape.size() == 2 && "Not a matrix");
	assert(mShape[0] == mShape[1] && "Not a square matrix");
	for (int i = 0; i < mShape[0]; i++)
	{
		operator()(i, i) = 1;
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
