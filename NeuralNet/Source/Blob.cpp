#include "Blob.h"

Blob::Blob()
{
	printf("WARNING: initializing blob with default contructor\n");
}

Blob::Blob(const TensorShape& shape) : Data(shape), Delta(shape)
{
}

Blob::Blob(Tensor data, Tensor delta) : Data(data), Delta(delta)
{
	assert(data.mSize == delta.mSize);
}

Blob::~Blob()
{
	Data.freemem();
	Delta.freemem();
}

void Blob::copyToGPU()
{
	Data.copyToGPU();
	Delta.copyToGPU();
}

void Blob::copyToCPU()
{
	Data.copyToCPU();
	Delta.copyToCPU();
}

void Blob::reshape(const TensorShape& shape)
{
	Data.mShape = shape;
	Delta.mShape = shape;
}

Blob* Blob::cut(uint64_t start, uint64_t len)
{
	return (new Blob(Data.cut(start, len), Delta.cut(start, len)));
}
