#include "math.h"
#include <vector>
#include "conio.h"
#include "assert.h"
#include "time.h"

typedef double Float;
typedef std::vector<Float> Vector;

double clamp(double x)
{
	return x > 0.99 ? 0.99 : (x < 0.01 ? 0.01 : x);
}

double sigmoid(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

class Neuron
{
public:
	Vector weights;
	double bias;
	double delta;
	double output;

	Neuron() {}
	Neuron(const Vector& w, double b) : weights(w), bias(b) {}
	Neuron(int num_weights) : weights(num_weights), bias(0.5) 
	{
		for (int i = 0; i < num_weights; i++)
		{
			//weights[i] = 0.0;
			weights[i] = (((rand() % 1024)-512)*1.0)/1024.0;
		}
	}
	~Neuron() {}

	double compute(const Vector& inputs)
	{
		assert(weights.size() == inputs.size());
		double res = bias;
		for (int i = 0; i < inputs.size(); i++)
		{
			res += weights[i] * inputs[i];
		}
		//printf("old output: %f\n", output);
		output = sigmoid(res);
		//printf("new output: %f\n", output);
		return output;
	}
};

class NeuralNet
{
public:
	std::vector<std::vector<Neuron>> Neurons;
	int input_size;

	NeuralNet() {}
	NeuralNet(int i_s) : input_size(i_s) {}
	~NeuralNet() {}

	void addLayer()
	{
		Neurons.push_back(std::vector<Neuron>(0));
	}

	void addNeuron(int layer)
	{
		if (layer == 0)
		{
			Neuron n(input_size);
			Neurons[layer].push_back(n);
		}
		else
		{
			Neuron n(Neurons[layer - 1].size());
			Neurons[layer].push_back(n);
		}
	}

	void addNeuron(Neuron n, int layer)
	{
		Neurons[layer].push_back(n);
	}

	Vector forward(Vector inputs)
	{
		for (int i = 0; i < Neurons.size(); i++)
		{
			Vector newinputs;
			for (int j = 0; j < Neurons[i].size(); j++)
			{
				newinputs.push_back(Neurons[i][j].compute(inputs));
			}
			inputs = newinputs;
		}
		return inputs;
	}

	double backprop(Vector input, Vector output)
	{
		Vector frwd = forward(input);
		//printf("forward: %f\n", frwd[0]);

		double error = 0.0;
		for (int i = 0; i < output.size(); i++)
		{
			double x = Neurons[Neurons.size() - 1][i].output;
			error += (output[i] - x)*(output[i] - x);
			Neurons[Neurons.size() - 1][i].delta = ((output[i] - x) * x * (1.0 - x));
			//printf("Desired output %d: %f, current output: %f\n", i, output[i], x);
			//printf("For neuron %d %d, setting delta %f\n", Neurons.size() - 1, i, Neurons[Neurons.size() - 1][i].delta);
		}
		//printf("\nError: %f\n\n", error);
		for (int i = 0; i < Neurons.size()-1; i++)
		{
			for (int j = 0; j < Neurons[i].size(); j++)
			{
				double delta = 0.0;
				for (int k = 0; k < Neurons[i + 1].size(); k++)
				{
					delta += Neurons[i + 1][k].weights[j] * Neurons[i + 1][k].delta;
					//delta += Neurons[i + 1][k].bias * Neurons[i + 1][k].delta;
				}

				double x = Neurons[i][j].output;
				Neurons[i][j].delta = (delta * x * (1.0 - x));
				//printf("For neuron %d %d, setting delta %f %f\n", i, j, Neurons[i][j].delta, x);
			}
		}
		//printf("\n");
		double alpha = 1;
		for (int i = 0; i < Neurons.size(); i++)
		{
			for (int j = 0; j < Neurons[i].size(); j++)
			{
				//printf("Updating weight for Neuron: %d %d\n", i, j);
				//printf("old bias: %f\n", Neurons[i][j].bias);
				Neurons[i][j].bias += alpha*Neurons[i][j].delta;
				//printf("new bias: %f\n", Neurons[i][j].bias);
				for (int k = 0; k < Neurons[i][j].weights.size(); k++)
				{
					double val = input[k];
					if (i > 0)
					{
						val = Neurons[i - 1][k].output;
					}
					//printf("input: %f, delta: %f\n", val, Neurons[i][j].delta);
					//printf("old weight: %f\n", Neurons[i][j].weights[k]);
					Neurons[i][j].weights[k] += alpha*val*Neurons[i][j].delta;
					//printf("new weight: %f\n", Neurons[i][j].weights[k]);
				}
				//printf("\n");
			}
		}
		return error;
	}
};

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
	nn.addLayer();
	for(int i = 0;i<10;i++)
		nn.addNeuron(0);
	//nn.addLayer();
	//for (int i = 0; i<1; i++)
	//	nn.addNeuron(1);
	nn.addLayer();
	nn.addNeuron(1);

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

	for (int j = 0; j < 1000; j++)
	{
		double error = 0.0;
		for (int i = 0; i < inputs.size(); i++)
		{
			error += nn.backprop(inputs[i], outputs[i]);
		}
		printf("FINAL ERROR: %f %d\n", error, j);
	}
	
	printf("%f %f %f %f\n", nn.forward(binaryrep(101,10))[0], nn.forward(binaryrep(103,10))[0], nn.forward(binaryrep(107,10))[0], nn.forward(binaryrep(109,10))[0]);
	printf("bd");
	
	_getch();

	return 0;
}