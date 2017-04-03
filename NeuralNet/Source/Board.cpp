#include "Board.h"

Board::Board() : mOptimizer(nullptr), mUseOptimizer(true)
{
}

Board::~Board()
{
	//Free memory
	for (size_t i = 0; i < mErrorFuncs.size(); i++)
	{
		delete mErrorFuncs[i];
	}
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		delete mNeurons[i];
	}
	for (size_t i = 0; i < mBlobs.size(); i++)
	{
		delete mBlobs[i];
	}
	delete mOptimizer;
}

void Board::addNeuron(Neuron* n)
{
	assert(mOptimizer != nullptr);
	mNeurons.push_back(n);
	auto variables = n->getVariables();
	for (size_t i = 0; i < variables.size(); i++)
	{
		mOptimizer->addVariable(variables[i]);
	}
}

Blob* Board::newBlob(const TensorShape& shape)
{
	Blob* b = new Blob(shape);
	mBlobs.push_back(b);
	return b;
}

void Board::addErrorFunction(ErrorFunction* err_func)
{
	mErrorFuncs.push_back(err_func);
}

void Board::setOptimizer(Optimizer* optimizer)
{
	mOptimizer = optimizer;
}

void Board::addPlaceholder(Tensor* placeholder)
{
	mPlaceholders.push_back(placeholder);
}

Tensor Board::forward(const Tensor& input)
{
	mNeurons[0]->mInput->Data.mData = input.mData;
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		mNeurons[i]->forward();
	}
	return mNeurons[mNeurons.size()-1]->mOutput->Data;
}

Float Board::backprop(const Tensor& input, const Tensor& output)
{
	clear_deltas();

	mNeurons[0]->mInput->Data.mData = input.mData;
	mErrorFuncs[0]->mTarget = &output;
	//Forward Pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		//printf("ff: %d\n", i);
		mNeurons[i]->forward();
	}

	//Calculate Error
	Float error = mErrorFuncs[0]->calculateError();

	//printf("backward\n");
	//Backward Pass
	for (int i = mNeurons.size() - 1; i >= 0; i--)
	{
		//printf("bb: %d\n", i);
		mNeurons[i]->backprop();
	}

	return error;
}

Float Board::backprop(const Tensor& input, const std::vector<Tensor*>& output)
{
	clear_deltas();

	mNeurons[0]->mInput->Data.mData = input.mData;

	for (size_t i = 0; i < mErrorFuncs.size(); i++)
	{
		mErrorFuncs[i]->mTarget = output[i];
	}
	
	//Forward Pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		//printf("ff: %d\n", i);
		mNeurons[i]->forward();
	}

	//Calculate Error
	Float error = 0;
	for (size_t i = 0; i < mErrorFuncs.size(); i++)
	{
		error += mErrorFuncs[i]->calculateError();
	}

	//printf("backward\n");
	//Backward Pass
	for (int i = mNeurons.size() - 1; i >= 0; i--)
	{
		//printf("bb: %d\n", i);
		mNeurons[i]->backprop();
	}

	return error;
}

Tensor Board::predict(const Tensor& input)
{
	mNeurons[0]->mInput->Data.mData = input.mData;

	//Forward Pass
	for (size_t i = 0; i < mNeurons.size(); i++)
	{
		mNeurons[i]->forward();
	}
	return mNeurons[mNeurons.size()-1]->mOutput->Data;
}

double Board::train(const Tensor& inputs, const Tensor& outputs, unsigned int epochs, unsigned int batch_size)
{
	assert(mErrorFuncs.size() > 0);
	//assert(mOptimizer != nullptr);
	assert(inputs.rows() == outputs.rows());
	assert(inputs.rows() % batch_size == 0);
	double error = 0.0;
	printf("Started training\n");
	Clock clock;
	clock.Start();
	for (int i = 0; i < epochs; i++)
	{
		error = 0.0;
		for (int j = 0; j < inputs.rows() / batch_size; j++)
		{
			error += backprop(inputs.cut(batch_size*j, batch_size), outputs.cut(batch_size*j, batch_size));

			if (mUseOptimizer)
				mOptimizer->optimize();
		}
		/*for (int i = 0; i < inputs.size(); i++)
		{
		error += backprop(inputs[i], outputs[i]);
		}*/
		clock.Stop();
		printf("Error %d: %f, epochs per sec: %f\n", i+1, error, ((i + 1)*1.0) / clock.ElapsedSeconds());
		printf("Batches per sec: %f\n", (i+1.0)*(inputs.rows()*1.0 / batch_size) / clock.ElapsedSeconds());
		//printf("Error %d: %f\n", j, error);
	}
	printf("Done training\n");
	return error;
}

void Board::save_variables(std::string filename)
{
	std::fstream file(filename, std::ios::out | std::ios::binary | std::ios::trunc);
	if (!file.is_open())
	{
		printf("unable to open file for saving: %s\n", filename.c_str());
		return;
	}
	for (size_t i = 0; i < mOptimizer->Variables.size(); i++)
	{
		for (uint64_t j = 0; j < mOptimizer->Variables[i]->Data.mSize; j++)
		{
			//file.write((const char*)&mBoard->mOptimizer->Variables[i]->Data.mData, sizeof(Float)*mBoard->mOptimizer->Variables[i]->Data.mSize);
			file.write((const char*)&mOptimizer->Variables[i]->Data(j), sizeof(Float));
			//file << mBoard->mOptimizer->Variables[i]->Data(j);
		}
		//file << "\n";
	}
	file.close();
}

void Board::load_variables(std::string filename)
{
	std::fstream file(filename, std::ios::in | std::ios::binary);
	if (!file.is_open())
	{
		printf("unable to open file for loading: %s\n", filename.c_str());
		return;
	}
	char* mem = new char[sizeof(Float)];
	for (size_t i = 0; i <mOptimizer->Variables.size(); i++)
	{
		for (uint64_t j = 0; j < mOptimizer->Variables[i]->Data.mSize; j++)
		{
			file.read(mem, sizeof(Float));
			memcpy(&mOptimizer->Variables[i]->Data(j), mem, sizeof(Float));
			//file >> mBoard->mOptimizer->Variables[i]->Data(j);
		}
		//file << "\n";
	}
	file.close();
}

void Board::copy_variables(const Board* b)
{
	for (size_t i = 0; i < mOptimizer->Variables.size(); i++)
	{
		memcpy(mOptimizer->Variables[i]->Data.mData, b->mOptimizer->Variables[i]->Data.mData, 
			sizeof(Float)*mOptimizer->Variables[i]->Data.mSize);
	}
}

void Board::clear_deltas()
{
	for (size_t i = 0; i < mBlobs.size(); i++)
	{
		mBlobs[i]->Delta.setzero();
	}
}
