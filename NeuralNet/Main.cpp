#include "Board.h"

#include <intrin.h>

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

Matrix openidx_input(std::string filename)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!(file.is_open()))
	{
		printf("unable to open file: %s", filename.c_str());
		return Matrix(0,0);
	}

	file.seekg(4, std::ios::beg);
	uint32_t num=0, row=0, col=0;
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
	Matrix result(row*col, num);
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
			result(j,i) = (tmp)/256.0;
			//printf("%f\n", result(j,i));
		}
		if(i%1000==0)
			printf("num: %d\n",i);
	}
	file.close();
	delete[] data_32;
	return result;
}

Matrix openidx_output(std::string filename, size_t output_size)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!(file.is_open()))
	{
		printf("unable to open file: %s", filename.c_str());
		return Matrix(0,0);
	}

	file.seekg(4, std::ios::beg);
	uint32_t num;
	char* data_32 = new char[4];
	file.read(data_32, 4);
	memcpy(&num, data_32, 4);
	num = _byteswap_ulong(num);
	//file >> num;
	printf("output size: %d\n", num);
	Matrix result(10,num);
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
				result(j,i) = 1.0;
			}
			else
			{
				result(j,i) = 0.0;
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
	Matrix inputs;
	Matrix outputs;

	TrainingData() {}
	TrainingData(Matrix i, Matrix o) : inputs(i), outputs(o) {}
};

TrainingData load_cifar(std::string filename)
{
	Matrix output(10, 10000);
	Matrix input(3072, 10000);
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
			input(j, i) = tmp/256.0;
		}

		if (i % 1000 == 0)
			printf("%d\n", i);
	}

	file.close();

	TrainingData td(input, output);
	return td;
}

void printinput(Vector input)
{
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			if (input[i * 28 + j] >= 0.1)
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

void printoutput(const Vector& output)
{
	for (int i = 0; i < output.size(); i++)
	{
		if (output[i] == 1)
		{
			printf("%d\n", i);
		}
	}
}

unsigned int getoutput(const Vector& output)
{
	assert(output.size() > 0);
	double max = output[0];
	unsigned int maxid = 0;
	for (size_t i = 1; i < output.size(); i++)
	{
		if (output[i] > max)
		{
			max = output[i];
			maxid = i;
		}
	}
	return maxid;
}

void test_old()
{
	/*Matrix inputs_train = openidx_input("Data/train-images.idx3-ubyte");
	Matrix outputs_train = openidx_output("Data/train-labels.idx1-ubyte", 10);
	Matrix inputs_test = openidx_input("Data/t10k-images.idx3-ubyte");
	Matrix outputs_test = openidx_output("Data/t10k-labels.idx1-ubyte",10);*/

	TrainingData b1 = load_cifar("Data/cifar-10-batches-bin/data_batch_1.bin");
	TrainingData b2 = load_cifar("Data/cifar-10-batches-bin/data_batch_2.bin");
	TrainingData b3 = load_cifar("Data/cifar-10-batches-bin/data_batch_3.bin");
	//TrainingData b4 = load_cifar("Data/cifar-10-batches-bin/data_batch_4.bin");
	//TrainingData b5 = load_cifar("Data/cifar-10-batches-bin/data_batch_5.bin");
	TrainingData b6 = load_cifar("Data/cifar-10-batches-bin/test_batch.bin");

	Matrix inputs_test = b6.inputs;
	Matrix outputs_test = b6.outputs;

	NeuralNetVectorized nn(b1.inputs.rows(), 0.01, 100);
	nn.addLayer(100);
	//nn.addLayer(100);
	//nn.addLayer(100);
	nn.addLayer(b1.outputs.rows());
	nn.randomizeWeights();

	//nn.load("net_handwriting.txt");

	nn.train(b1.inputs, b1.outputs, 10);
	nn.train(b2.inputs, b2.outputs, 10);
	nn.train(b3.inputs, b3.outputs, 10);
	//nn.train(b4.inputs, b4.outputs, 10);
	//nn.train(b5.inputs, b5.outputs, 10);

	int acc = 0;
	for (size_t i = 0; i < inputs_test.cols(); i++)
	{
		if (getoutput(nn.predict(inputs_test.col(i))) == getoutput(outputs_test.col(i)))
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

int main()
{
	srand(time(0));

	Board b;

	Blob* inputBlob = b.newBlob();
	Blob* layer1Blob = b.newBlob();
	Blob* outputBlob = b.newBlob();
	b.addNeuron(new SigmoidNeuron(inputBlob, layer1Blob, 0.1));
	b.addNeuron(new SigmoidNeuron(layer1Blob, outputBlob, 0.1));
	b.setErrorFunction(new MeanSquaredError(inputBlob, outputBlob, nullptr));

	TrainingData b1 = load_cifar("Data/cifar-10-batches-bin/data_batch_1.bin");
	TrainingData b2 = load_cifar("Data/cifar-10-batches-bin/data_batch_2.bin");
	TrainingData b3 = load_cifar("Data/cifar-10-batches-bin/data_batch_3.bin");

	TrainingData b6 = load_cifar("Data/cifar-10-batches-bin/test_batch.bin");

	Matrix inputs_test = b6.inputs;
	Matrix outputs_test = b6.outputs;

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

	return 0;
}