#include "Blob.h"

Blob::Blob()
{
	printf("WARNING: initializing blob with default contructor\n");
}

Blob::Blob(const TensorShape& shape) : Data(shape), Delta(shape)
{
}

Blob::~Blob()
{
	Data.freememory();
	Delta.freememory();
}

void Blob::reshape(const TensorShape& shape)
{
	Data.mShape = shape;
	Delta.mShape = shape;
}
