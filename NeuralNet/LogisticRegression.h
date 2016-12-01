#pragma once

#include "Typedefs.h"
#include "Clock.h"

class LogisticRegression
{
public:
	size_t InputSize;
	size_t OutputSize;
	std::vector<Vector> Weights;
	Vector Bias;

	Float LearningRate;

	LogisticRegression();
	LogisticRegression(size_t InputSize, size_t OutputSize, double learning_rate);
	~LogisticRegression();

	Vector calculate(Vector input);
	void learn(Vector input, Vector output);
	double error(Dataset inputs, Dataset outputs);
	double train(Dataset inputs, Dataset outputs, unsigned int epochs);
	
	void save(std::string filename) const;
	void load(std::string filename);
};

double clamp(double x);

inline double sigmoid(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

inline double rand_init()
{
	return ((rand() % 1024) / 1024.0) - 0.5;
}
