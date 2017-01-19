#include "Tensor.h"

Tensor::Tensor()
{
}

Tensor::Tensor(unsigned int a) : t(a)
{
	t(3, 3, 4);
}

Tensor::Tensor(unsigned int a, unsigned int b) : t(a,b)
{
}

Tensor::Tensor(unsigned int a, unsigned int b, unsigned int c) : t(a,b,c)
{
}

Tensor::Tensor(unsigned int a, unsigned int b, unsigned int c, unsigned int d) : t(a,b,c,d)
{
}

Tensor::~Tensor()
{
}
