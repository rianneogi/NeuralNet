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
