#include "NeuralNet.h"

Vector binaryrep(int x, int size)
{
	Vector v;
	int cnt = 0;
	while (x != 0)
	{
		v.push_back(x % 2);
		x /= 2;
		cnt++;
	}
	while (cnt < size)
	{
		v.push_back(0);
		cnt++;
	}
	return v;
}
 
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

int main()
{
	srand(time(0));

	NeuralNet nn(10);
	//nn.addLayer();
	//for(int i = 0;i<5;i++)
	//	nn.addNeuron(0);
	////nn.addLayer();
	////for (int i = 0; i<1; i++)
	////	nn.addNeuron(1);
	//nn.addLayer();
	//nn.addNeuron(1);

	nn.load("net.txt");

	std::vector<int> primes = genprimes(100);
	std::vector<Vector> inputs;
	std::vector<Vector> outputs;
	for (int i = 0; i < 100; i++)
	{
		inputs.push_back(binaryrep(i, 10));
		if (isprime(i))
		{
			Vector v;
			v.push_back(1.0);
			outputs.push_back(v);
		}
		else
		{
			Vector v;
			v.push_back(0.0);
			outputs.push_back(v);
		}
	}

	nn.train(inputs, outputs, 100);
	
	printf("%f %f %f %f\n", nn.forward(binaryrep(101,10))[0], nn.forward(binaryrep(103,10))[0], nn.forward(binaryrep(107,10))[0], nn.forward(binaryrep(109,10))[0]);
	printf("bd");

	nn.save("net.txt");
	
	_getch();

	return 0;
}