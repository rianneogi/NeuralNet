#include "Tests.h"

int main()
{
	srand(time(0));

	//test_new();

	Tensor t(make_shape(2,2));
	printf("%f\n", t(0, 0));
	t.print();
	t.setidentity();
	printf("%f\n", t(0, 0));
	t.print();
	t(0, 0) = 20;
	printf("%f\n", t(0, 0));
	t.print();
	_getch();

	return 0;
}