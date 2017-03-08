#include <intrin.h>
#include "Tests.h"

//Vector binaryrep(int x, int size)
//{
//	Vector v;
//	int cnt = 0;
//	while (x != 0)
//	{
//		v.push_back(x % 2);
//		x /= 2;
//		cnt++;
//	}
//	while (cnt < size)
//	{
//		v.push_back(0);
//		cnt++;
//	}
//	return v;
//}

bool isprime(int x)
{
	if (x == 1 || x == 0) return false;
	int sq = sqrt(x);
	for (int i = 2; i <= sq; i++)
	{
		if (x%i == 0)
			return false;
	}
	return true;
}

std::vector<int> genprimes(int num)
{
	std::vector<int> primes;
	int cnt = 0;
	for (int i = 2; cnt < num; i++)
	{
		if (isprime(i))
		{
			primes.push_back(i);
			cnt++;
		}
	}
	return primes;
}

Tensor openidx_input(std::string filename)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!(file.is_open()))
	{
		printf("unable to open file: %s", filename.c_str());
		return Tensor();
	}

	file.seekg(4, std::ios::beg);
	uint32_t num = 0, row = 0, col = 0;
	char* data_32 = new char[4];

	file.read(data_32, 4);
	memcpy(&num, data_32, 4);
	num = _byteswap_ulong(num);

	file.read(data_32, 4);
	memcpy(&row, data_32, 4);
	row = _byteswap_ulong(row);

	file.read(data_32, 4);
	memcpy(&col, data_32, 4);
	col = _byteswap_ulong(col);

	//file >> num >> row >> col;
	printf("input size: %d %d %d\n", num, row, col);
	Tensor result(make_shape(row*col, num));
	unsigned char tmp;
	char byte;
	for (size_t i = 0; i < num; i++)
	{
		//result.push_back(Vector(row*col));
		for (size_t j = 0; j < row*col; j++)
		{
			//file >> byte;
			file.read(&byte, 1);
			memcpy(&tmp, &byte, 1);
			//byte = _byteswap_ushort(byte);
			//result[i][j] = tmp;
			result(j, i) = (tmp) / 256.0;
			//printf("%f\n", result(j,i));
		}
		if (i % 1000 == 0)
			printf("num: %d\n", i);
	}
	file.close();
	delete[] data_32;
	return result;
}

Tensor openidx_output(std::string filename, size_t output_size)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!(file.is_open()))
	{
		printf("unable to open file: %s", filename.c_str());
		return Tensor();
	}

	file.seekg(4, std::ios::beg);
	uint32_t num;
	char* data_32 = new char[4];
	file.read(data_32, 4);
	memcpy(&num, data_32, 4);
	num = _byteswap_ulong(num);
	//file >> num;
	printf("output size: %d\n", num);
	Tensor result(make_shape(10, num));
	for (size_t i = 0; i < num; i++)
	{
		//result.push_back(Vector(output_size));
		char byte;
		file.read(&byte, 1);
		//file >> byte;
		for (size_t j = 0; j < output_size; j++)
		{
			if (j == byte)
			{
				result(j, i) = 1.0;
			}
			else
			{
				result(j, i) = 0.0;
			}
		}
		if (i % 1000 == 0)
			printf("num: %d\n", i);
	}
	file.close();
	delete[] data_32;
	return result;
}

struct TrainingData
{
	Tensor inputs;
	Tensor outputs;

	TrainingData() {}
	TrainingData(Tensor i, Tensor o) : inputs(i), outputs(o) {}
};

TrainingData load_cifar(std::string filename)
{
	Tensor output(make_shape(10, 10000));
	Tensor input(make_shape(3072, 10000));
	std::fstream file(filename, std::ios::in | std::ios::binary);

	if (!file.is_open())
	{
		printf("cant open file: %s\n", filename);
		return TrainingData();
	}

	unsigned char tmp;
	char byte;
	for (int i = 0; i < 10000; i++)
	{
		file.read(&byte, 1);
		for (int j = 0; j < 10; j++)
		{
			if (byte == j)
			{
				output(j, i) = 1.0;
			}
			else
			{
				output(j, i) = 0.0;
			}
		}

		for (int j = 0; j < 3072; j++)
		{
			file.read(&byte, 1);
			memcpy(&tmp, &byte, 1);
			input(j, i) = tmp / 256.0;
		}

		if (i % 1000 == 0)
			printf("%d\n", i);
	}

	file.close();

	TrainingData td(input, output);
	return td;
}

void printinput(Tensor input)
{
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			if (input(i * 28 + j) >= 0.1)
			{
				printf("1");
			}
			else
			{
				printf("0");
			}
		}
		printf("\n");
	}
}

void printoutput(const Tensor& output)
{
	for (int i = 0; i < output.mSize; i++)
	{
		if (output(i) == 1)
		{
			printf("%d\n", i);
		}
	}
}

unsigned int getoutput(const Tensor& output)
{
	assert(output.mSize > 0);
	double max = output(0);
	unsigned int maxid = 0;
	for (size_t i = 1; i < output.mSize; i++)
	{
		if (output(i) > max)
		{
			max = output(i);
			maxid = i;
		}
	}
	return maxid;
}

void test_old()
{
	///*Matrix inputs_train = openidx_input("Data/train-images.idx3-ubyte");
	//Matrix outputs_train = openidx_output("Data/train-labels.idx1-ubyte", 10);
	//Matrix inputs_test = openidx_input("Data/t10k-images.idx3-ubyte");
	//Matrix outputs_test = openidx_output("Data/t10k-labels.idx1-ubyte",10);*/

	//TrainingData b1 = load_cifar("Data/cifar-10-batches-bin/data_batch_1.bin");
	//TrainingData b2 = load_cifar("Data/cifar-10-batches-bin/data_batch_2.bin");
	//TrainingData b3 = load_cifar("Data/cifar-10-batches-bin/data_batch_3.bin");
	////TrainingData b4 = load_cifar("Data/cifar-10-batches-bin/data_batch_4.bin");
	////TrainingData b5 = load_cifar("Data/cifar-10-batches-bin/data_batch_5.bin");
	//TrainingData b6 = load_cifar("Data/cifar-10-batches-bin/test_batch.bin");

	//Matrix inputs_test = b6.inputs;
	//Matrix outputs_test = b6.outputs;

	//NeuralNetVectorized nn(b1.inputs.rows(), 0.01, 100);
	//nn.addLayer(100);
	////nn.addLayer(100);
	////nn.addLayer(100);
	//nn.addLayer(b1.outputs.rows());
	//nn.randomizeWeights();

	////nn.load("net_handwriting.txt");

	//nn.train(b1.inputs, b1.outputs, 10);
	//nn.train(b2.inputs, b2.outputs, 10);
	//nn.train(b3.inputs, b3.outputs, 10);
	////nn.train(b4.inputs, b4.outputs, 10);
	////nn.train(b5.inputs, b5.outputs, 10);

	//int acc = 0;
	//for (size_t i = 0; i < inputs_test.cols(); i++)
	//{
	//	if (getoutput(nn.predict(inputs_test.col(i))) == getoutput(outputs_test.col(i)))
	//	{
	//		acc++;
	//	}
	//	/*else
	//	{
	//	printinput(inputs_train.col(i));
	//	printoutput(nn.forward(inputs_train.col(i)));
	//	printf("%d %d\n", getoutput(nn.forward(inputs_train.col(i))), getoutput(outputs_train.col(i)));
	//	}*/
	//}
	//printf("Accuracy: %f\n", (acc*1.0) / inputs_test.cols());

	////nn.save("net_handwriting.txt");
	//_getch();
}

void test_new()
{
	//MNIST input size: 28x28 = 784
	//CIFAR input size: 3072

	Board b;
	int batch_size = 100;
	double learning_rate = 0.005;

	Blob* inputBlob = b.newBlob(784, batch_size);
	Blob* layer1FCBlob = b.newBlob(14, batch_size);
	Blob* layer1SigBlob = b.newBlob(14, batch_size);
	Blob* layer2FCBlob = b.newBlob(12, batch_size);
	Blob* layer2SigBlob = b.newBlob(12, batch_size);
	Blob* outputFCBlob = b.newBlob(10, batch_size);
	Blob* outputSigBlob = b.newBlob(10, batch_size);
	b.addNeuron(new FullyConnectedNeuron(inputBlob, layer1FCBlob, learning_rate));
	b.addNeuron(new TanhNeuron(layer1FCBlob, layer1SigBlob, learning_rate));
	b.addNeuron(new FullyConnectedNeuron(layer1SigBlob, layer2FCBlob, learning_rate));
	b.addNeuron(new TanhNeuron(layer2FCBlob, layer2SigBlob, learning_rate));
	b.addNeuron(new FullyConnectedNeuron(layer2SigBlob, outputFCBlob, learning_rate));
	b.addNeuron(new TanhNeuron(outputFCBlob, outputSigBlob, learning_rate));
	b.setErrorFunction(new MeanSquaredError(inputBlob, outputSigBlob, nullptr));

	Tensor inputs_train = openidx_input("Data/train-images.idx3-ubyte");
	Tensor outputs_train = openidx_output("Data/train-labels.idx1-ubyte", 10);
	Tensor inputs_test = openidx_input("Data/t10k-images.idx3-ubyte");
	Tensor outputs_test = openidx_output("Data/t10k-labels.idx1-ubyte", 10);

	/*TrainingData b1 = load_cifar("Data/cifar-10-batches-bin/data_batch_1.bin");
	TrainingData b2 = load_cifar("Data/cifar-10-batches-bin/data_batch_2.bin");
	TrainingData b3 = load_cifar("Data/cifar-10-batches-bin/data_batch_3.bin");

	TrainingData b6 = load_cifar("Data/cifar-10-batches-bin/test_batch.bin");

	Matrix inputs_test = b6.inputs;
	Matrix outputs_test = b6.outputs;*/


	b.train(inputs_train, outputs_train, 10, 100);

	/*for (int i = 0; i < 10; i++)
	{
	b.train(b1.inputs, b1.outputs, 1, 100);
	b.train(b2.inputs, b2.outputs, 1, 100);
	b.train(b3.inputs, b3.outputs, 1, 100);
	}*/


	int acc = 0;
	for (size_t i = 0; i < inputs_test.cols(); i++)
	{
		if (getoutput(b.predict(inputs_test.col(i))) == getoutput(outputs_test.col(i)))
		{
			acc++;
		}
		/*else
		{
		printinput(inputs_train.col(i));
		printoutput(nn.forward(inputs_train.col(i)));
		printf("%d %d\n", getoutput(nn.forward(inputs_train.col(i))), getoutput(outputs_train.col(i)));
		}*/
	}
	printf("Accuracy: %f\n", (acc*1.0) / inputs_test.cols());

	//nn.save("net_handwriting.txt");
	_getch();
}

void test_dgemm()
{
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
}
