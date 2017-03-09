#include "Blob.h"

Blob::Blob()
{
	printf("WARNING: initializing blob with default contructor\n");
}

Blob::Blob(unsigned int rows, unsigned int cols) : Data(make_shape(rows, cols)), Delta(make_shape(rows, cols))
{
}

Blob::~Blob()
{
	Data.freememory();
	Delta.freememory();
}
