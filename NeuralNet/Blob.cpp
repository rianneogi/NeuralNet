#include "Blob.h"

Blob::Blob()
{
	printf("WARNING: initializing blob with default contructor\n");
}

Blob::Blob(unsigned int rows, unsigned int cols) : Data(rows, cols), Delta(rows, cols)
{
}

Blob::~Blob()
{
}
