#include "Tests.h"

int main()
{
	srand(time(0));

	//test_new();

	Tensor t1(make_shape(2, 3));
	t1(0, 0) = -1;
	t1(0, 1) = 1;
	t1(0, 2) = 4;
	t1(1, 0) = -4;
	t1(1, 1) = 0;
	t1(1, 2) = -3;
	t1.print();

	Tensor t2(make_shape(3, 4));
	t2(0, 0) = 2;
	t2(0, 1) = 3;
	t2(0, 2) = -2;
	t2(0, 3) = 1;
	t2(1, 0) = 4;
	t2(1, 1) = 0;
	t2(1, 2) = 5;
	t2(1, 3) = 6;
	t2(2, 0) = 7;
	t2(2, 1) = 8;
	t2(2, 2) = 9;
	t2(2, 3) = 10;
	t2.print();

	Tensor t3(make_shape(2, 4));

	//Mat Mul
	/*clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, t1.cols(), t2.rows(),
		t1.rows(), 1, t1.mData, t1.rows(), t2.mData, t2.rows(), 0, t3.mData, t3.rows())*/
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, t1.cols(), t2.rows(),
		t1.rows(), 1, t1.mData, t1.rows(), t2.mData, t2.rows(), 0, t3.mData, t3.rows());
	t3.print();

	_getch();

	return 0;
}