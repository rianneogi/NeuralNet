#include "LogisticRegression.h"

double clamp(double x)
{
	return x > 1.0 ? 1.0 : (x < 0.0 ? 0.0 : x);
}

double sigmoid(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

double rand_init()
{
	return ((rand() % 1024) / 1024.0) - 0.5;
}

LogisticRegression::LogisticRegression()
{
}

LogisticRegression::LogisticRegression(size_t input_size, size_t output_size, double learning_rate) : InputSize(input_size), OutputSize(output_size), 
Bias(output_size), Weights(output_size), LearningRate(learning_rate)
{
	for (size_t i = 0; i < OutputSize; i++)
	{
		Bias[i] = rand_init();
		Weights[i] = Vector(InputSize);
		for (size_t j = 0; j < InputSize; j++)
		{
			Weights[i][j] = rand_init();
		}
	}
}

LogisticRegression::~LogisticRegression()
{
}

Vector LogisticRegression::calculate(Vector input)
{
	Vector v(OutputSize);
	for (size_t i = 0; i < OutputSize; i++)
	{
		double sum = Bias[i];
		for (size_t j = 0; j < InputSize; j++)
		{
			sum += input[j] * Weights[i][j];
		}
		v[i] = sigmoid(sum);
	}
	return v;
}

void LogisticRegression::learn(Vector input, Vector target)
{
	// Error = sum[0.5*(output[i]-target[i])^2]/n
	//       = sum[0.5*((sig(bias+w1i1+....wnin)-target[i])^2]/n
	// Gradient = (sig(bias+w1i1+w2+i2...wnin)-target[i])*d sig(bias+w1i1+w2+i2...wnin)
	//          = (sig(bias+w1i1+w2+i2...wnin)-target[i])*sig(bias+w1i1+w2+i2...wnin)*(1-sig(bias+w1i1+w2+i2...wnin))
	assert(target.size() == OutputSize);
	assert(input.size() == InputSize);
	Vector output = calculate(input);
	for (size_t i = 0; i < OutputSize; i++)
	{
		double d_bias = (output[i] - target[i]);
		Bias[i] -= LearningRate*d_bias;
		for (size_t j = 0; j < InputSize; j++)
		{
			double d_w = (output[i] - target[i])*input[j];
			Weights[i][j] -= LearningRate*d_w;
		}
	}
}

double LogisticRegression::error(Dataset inputs, Dataset outputs)
{
	double err = 0.0;
	double maxtmp = 0.0;
	//printf("error");
	for (size_t i = 0; i < inputs.size(); i++)
	{
		Vector output = calculate(inputs[i]);
		for (size_t j = 0; j < OutputSize; j++)
		{
			//err += 0.5*(outputs[i][j] - output[j])*(outputs[i][j] - output[j]);
			//printf("%f\n", err);
			double op = output[j];
			if (op <= 0.0)
				op = 0.0001;
			if (op >= 1.0)
				op = 0.9999;
			//printf("%f %f %f\n", op, log(op), log(1.0-op));
			double tmp = -outputs[i][j] * log(op) - (1.0 - outputs[i][j])*log(1.0 - op);
			if (tmp > maxtmp)
				maxtmp = tmp;
			//printf("%f\n", tmp);
			err += tmp;
			//printf("%f", err);
		}
	}
	printf("maxtmp: %f\n", maxtmp);
	return (err);
}

double LogisticRegression::train(Dataset inputs, Dataset outputs, unsigned int epochs)
{
	assert(inputs.size() == outputs.size());
	double err = 0.0;
	printf("Started training\n");
	Clock clock;
	clock.Start();
	for (unsigned int j = 0; j < epochs; j++)
	{
		err = error(inputs, outputs);
		clock.Stop();
		printf("Error %d: %f, epochs per sec: %f\n", j, err, (j*1.0)/clock.ElapsedSeconds());
		for (size_t i = 0; i < inputs.size(); i++)
		{
			learn(inputs[i], outputs[i]);
		}
	}
	printf("Done training\n");
	return err;
}