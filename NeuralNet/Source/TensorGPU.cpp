#include "TensorGPU.h"

TensorGPU::TensorGPU() : mData(NULL), mSize(0)
{
}

TensorGPU::TensorGPU(const TensorShape& shape) : mData(NULL), mShape(shape), mSize(1)
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	for (unsigned int x : mShape)
	{
		mSize *= x;
	}
	//printf("Size : %d\n", mSize);
	allocate();
}

TensorGPU::TensorGPU(Float* data, const TensorShape& shape) : mData(data), mShape(shape), mSize(1)
{
	assert(shape.size() <= 4 && "Max supported tensor shape is 4");
	for (unsigned int x : mShape)
	{
		mSize *= x;
	}
}

TensorGPU::~TensorGPU()
{
	//if(mSelfAllocated)
	//	freememory();
}

TensorGPU::TensorGPU(const TensorGPU& other) : mData(other.mData), mShape(other.mShape), mSize(other.mSize)
{
}

Float& TensorGPU::operator()(uint64_t a) const
{
#ifdef NN_DEBUG
	assert(a < mSize);
#endif
	return mData[a];
}

Float& TensorGPU::operator()(uint64_t a, uint64_t b) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 1);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
#endif
	return mData[a*mShape[1] + b];
}

Float& TensorGPU::operator()(uint64_t a, uint64_t b, uint64_t c) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 2);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
	assert(c < mShape[2]);
#endif
	return mData[a*mShape[1] * mShape[2] + b*mShape[2] + c];
}

Float& TensorGPU::operator()(uint64_t a, uint64_t b, uint64_t c, uint64_t d) const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 3);
	assert(a < mShape[0]);
	assert(b < mShape[1]);
	assert(c < mShape[2]);
	assert(d < mShape[3]);
#endif
	return mData[a*mShape[1] * mShape[2] * mShape[3] + b*mShape[2] * mShape[3] + c*mShape[3] + d];
}

void TensorGPU::allocate()
{
	//printf("Allocation tensor of size: %d\n", mSize);
#ifdef USE_MALLOC
	mData = (Float*)malloc(mSize * sizeof(Float));
	if (mData == NULL)
	{
		printf("ERROR: Cant allocate memory for tensor, Size: %d\n", mSize);
	}
#else
	mData = new Float[mSize];
#endif
}

void TensorGPU::freememory()
{
	if (mData != NULL)
	{
		//printf("Freeing memory: %d\n", mSize);
#ifdef USE_MALLOC
		free(mData);
		mData = NULL;
#else
		delete[] mData;
		mData = NULL;
#endif
	}
}

void TensorGPU::setzero()
{
	memset(mData, 0, sizeof(Float)*mSize);
}

void TensorGPU::setconstant(Float c)
{
	for (int i = 0; i < mSize; i++)
	{
		mData[i] = c;
	}
}

void TensorGPU::setidentity()
{
	setzero();
	assert(mShape.size() == 2 && "Not a matrix");
	assert(mShape[0] == mShape[1] && "Not a square matrix");
	for (int i = 0; i < mShape[0]; i++)
	{
		operator()(i, i) = 1;
	}
}

TensorGPU TensorGPU::cut(uint64_t begin, uint64_t len) const
{
	//printf("%d %d %d\n", begin, len, mShape[0]);
#ifdef NN_DEBUG
	assert(begin + len <= mShape[0]);
#endif
	TensorShape shape = mShape;
	shape[0] = len;
	return TensorGPU(&mData[begin*(mSize / mShape[0])], shape);
}

uint64_t TensorGPU::rows() const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 1);
#endif
	return mShape[0];
}

uint64_t TensorGPU::cols() const
{
#ifdef NN_DEBUG
	assert(mShape.size() >= 2);
#endif
	return mShape[1];
}

void TensorGPU::print() const
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

void TensorGPU::printshape() const
{
	for (int i = 0; i < mShape.size(); i++)
	{
		printf("%d ", mShape[i]);
	}
	printf("\n");
}