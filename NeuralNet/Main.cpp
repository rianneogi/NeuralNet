#include "NeuralNetVectorized.h"

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

std::vector<Vector> openidx_input(std::string filename)
{
	std::vector<Vector> result;

	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!(file.is_open()))
	{
		printf("unable to open file: %s", filename.c_str());
		return result;
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
	unsigned char tmp;
	char byte;
	for (size_t i = 0; i < num; i++)
	{
		result.push_back(Vector(row*col));
		for (size_t j = 0; j < row*col; j++)
		{
			//file >> byte;
			file.read(&byte, 1);
			memcpy(&tmp, &byte, 1);
			//byte = _byteswap_ushort(byte);
			//result[i][j] = tmp;
			result[i][j] = (tmp)/256.0;
			//printf("%f\n", result[i][j]);
		}
		if(i%1000==0)
			printf("num: %d\n",i);
	}
	
	delete[] data_32;
	return result;
}

std::vector<Vector> openidx_output(std::string filename, size_t output_size)
{
	std::vector<Vector> result;

	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!(file.is_open()))
	{
		printf("unable to open file: %s", filename.c_str());
		return result;
	}

	file.seekg(4, std::ios::beg);
	uint32_t num;
	char* data_32 = new char[4];
	file.read(data_32, 4);
	memcpy(&num, data_32, 4);
	num = _byteswap_ulong(num);
	//file >> num;
	printf("output size: %d\n", num);
	for (size_t i = 0; i < num; i++)
	{
		result.push_back(Vector(output_size));
		char byte;
		file.read(&byte, 1);
		//file >> byte;
		for (size_t j = 0; j < output_size; j++)
		{
			if (j == byte)
			{
				result[i][j] = 1.0;
			}
			else
			{
				result[i][j] = 0.0;
			}
		}
		if (i % 1000 == 0)
			printf("num: %d\n", i);
	}
	delete[] data_32;
	return result;
}

void printinput(Vector input)
{
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			if (input[i * 28 + j] >= 0)
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

void printoutput(Vector output)
{
	for (int i = 0; i < output.size(); i++)
	{
		if (output[i] == 1)
		{
			printf("%d\n", i);
		}
	}
}

unsigned int getoutput(Vector output)
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

int main()
{
	srand(time(0));

	auto inputs = openidx_input("Data/train-images.idx3-ubyte");
	auto outputs = openidx_output("Data/train-labels.idx1-ubyte", 10);
	auto inputs_train = inputs;
	auto outputs_train = outputs;
	std::vector<Vector> inputs_test, outputs_test;

	for (int i = 0; i < 10000; i++)
	{
		inputs_test.push_back(inputs_train[inputs_train.size() - 1]);
		inputs_train.pop_back();
	}
	for (int i = 0; i < 10000; i++)
	{
		outputs_test.push_back(outputs_train[outputs_train.size() - 1]);
		outputs_train.pop_back();
	}

	//LogisticRegression lr(28 * 28, 10, 0.00001);
	//lr.train(inputs_train, outputs_train, 100);
	//printf("FINAL ERROR: %f\n", lr.error(inputs_test, outputs_test));

	//double acc = 0.0;
	//for (size_t i = 0; i < inputs.size(); i++)
	//{
	//	if (getoutput(lr.calculate(inputs[i])) == getoutput(outputs[i]))
	//	{
	//		acc+=1;
	//	}
	//	/*else
	//	{
	//		printinput(inputs[i]);
	//		printoutput(lr.calculate(inputs[i]));
	//		printf("%d %d\n", getoutput(lr.calculate(inputs[i])), getoutput(outputs[i]));
	//		for (int j = 0; j < outputs[i].size(); j++)
	//		{
	//			printf("%f ", lr.calculate(inputs[i])[j]);
	//		}
	//		printf("\n");
	//		for (int j = 0; j < outputs[i].size(); j++)
	//		{
	//			printf("%f ", outputs[i][j]);
	//		}
	//		printf("\n");
	//	}*/
	//}
	//printf("acc: %f\n", acc / inputs.size());

	NeuralNetVectorized nn(inputs[0].size(), 1.0);
	/*nn.addLayer();
	for(int i = 0;i<15;i++)
		nn.addNeuron(0);
	nn.addLayer();
	for (int i = 0; i<outputs[0].size(); i++)
		nn.addNeuron(1);*/

	
	nn.load("net_handwriting.txt");

	nn.train(inputs_train, outputs_train, 10);
	
	int acc = 0;
	for (size_t i = 0; i < inputs_test.size(); i++)
	{
		if (getoutput(nn.forward(inputs_test[i])) == getoutput(outputs_test[i]))
		{
			acc++;
		}
		/*else
		{
			printinput(inputs[i]);
			printoutput(nn.forward(inputs[i]));
			printf("%d %d\n", getoutput(nn.forward(inputs[i])), getoutput(outputs[i]));
		}*/
	}
	printf("Accuracy: %f\n", (acc*1.0) / inputs_test.size());

	//nn.save("net_handwriting.txt");
	
	_getch();

	return 0;
}