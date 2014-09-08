//Student:			Adam Sundstrom
//Student Number:	SNDADA001
//Assignment:		Back Propagation

//Includes
#include "CNeuralNet.h"
#include <iostream>

//Struct to represent a layer of neurons
//Takes in the number of neurons, and the number of imports that each neuron will receive (to construct the neurons)
SNeuronLayer::SNeuronLayer(int NumNeurons, int NumInputsPerNeuron) : m_iNumNeurons(NumNeurons)
{
	for (int i = 0; i<NumNeurons; ++i)

		m_vecNeurons.push_back(SNeuron(NumInputsPerNeuron));
}

//Struct to represent an individual neuron
//Takes in the number of inputs that the neuron will receive
SNeuron::SNeuron(int NumInputs) : m_iNumInputs(NumInputs), m_Activation(0), m_Error(0)
{
	//initialize the weights
	for (int i = 0; i<NumInputs; ++i)
	{
		m_vecWeight.push_back(0);	//initialized weights to 0 since they are overwritten later
		m_prevWeight.push_back(0);
	}
}


 //The constructor of the neural network. This constructor will allocate memory
 //for the weights of both input->hidden and hidden->output layers, as well as the input, hidden
 //and output layers.

CNeuralNet::CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff) 
: m_inputLayerSize(inputLayerSize), m_hiddenLayerSize(hiddenLayerSize), m_outputLayerSize(outputLayerSize), m_lRate(lRate), m_mse_cutoff(mse_cutoff) //intializer list
{
	//initialize vectors for output layer values
	std::vector<double> _outputActivation(m_outputLayerSize);

	//initialize the two neuron layers
	SNeuronLayer hiddenLayer(m_hiddenLayerSize, m_inputLayerSize);
	SNeuronLayer outputLayer(m_outputLayerSize, m_hiddenLayerSize);

	//push the layers onto a vector
	m_vecLayer.push_back(hiddenLayer);
	m_vecLayer.push_back(outputLayer);

	//initialize random weights for neurons
	initWeights();

	//std::cout << "Constructor complete" << std::endl;
}

 //The destructor of the class. All allocated memory will be released here
CNeuralNet::~CNeuralNet() 
{
		//no memory to free up
}

//Method to initialize the both layers of weights to random numbers
void CNeuralNet::initWeights()
{
	for (int i = 0; i < 2; ++i) //iterate through the two layers (hidden and output layers)
	{
		for (int j = 0; j < m_vecLayer[i].m_iNumNeurons; ++j) //iterate through the neurons in each layer
		{
			for (int k = 0; k < m_vecLayer[i].m_vecNeurons[j].m_iNumInputs; ++k) //iterate through each input weight going into each neuron
			{ 
				//generate a random float between -1 and 1
				m_vecLayer[i].m_vecNeurons[j].m_vecWeight[k] = RandomClamped();;	//assign the weight
			}
		}
	}
	//std::cout << "Init Weights complete" << std::endl;
}

//Basic Sigmoid function
double CNeuralNet::Sigmoid(double netinput)
{
	return (1 / (1 + exp(-netinput)));
}

//This is the forward feeding part of back propagation.
//1. This should take the input and copy the memory (use memcpy / std::copy)
//to the allocated _input array.
//2. Compute the output of at the hidden layer nodes 
//(each _hidden layer node = sigmoid (sum( _weights_h_i * _inputs)) //assume the network is completely connected
//3. Repeat step 2, but this time compute the output at the output layer

void CNeuralNet::feedForward(std::vector<double> inputs)
{
	std::vector<double> _inputs = inputs;	//store inputs in reference vector
	

	vector<double> outputs; //store the result of the outputs from each layer

	//For both layers (hidden and output)  
	for (int i = 0; i < 2; ++i)
	{

		if (i > 0)				//checks which layer you're on
		{
			_inputs = outputs;	//sets inputs to outputs if you're past the 'first' layer
		}

		outputs.clear();

	//iterate through neurons and get the sum of weights * inputs  
		for (int n = 0; n < m_vecLayer[i].m_iNumNeurons; ++n)
		{
			double netinput = 0;

			int NumInputs = m_vecLayer[i].m_vecNeurons[n].m_iNumInputs;

			//for each weight   
			for (int k = 0; k < NumInputs; ++k)
			{
				//sum the weights x inputs   
				netinput += m_vecLayer[i].m_vecNeurons[n].m_vecWeight[k] * _inputs[k];
			}

			//The combined activation is first filtered through the sigmoid    
			//function and a record is kept for each neuron    
			m_vecLayer[i].m_vecNeurons[n].m_Activation = Sigmoid(netinput);

			//store the outputs from each layer as we generate them.   
			outputs.push_back(m_vecLayer[i].m_vecNeurons[n].m_Activation);
		}
	}
	_outputActivation = outputs; //save what was in the output layer for later reference
	//std::cout << "Feedforward complete" << std::endl;
}

/**
 This is the actual back propagation part of the back propagation algorithm
 It should be executed after feeding forward. Given a vector of desired outputs
 we compute the error at the hidden and output layers (allocate some memory for this) and
 assign 'blame' for any error to all the nodes that fed into the current node, based on the
 weight of the connection.
 Steps:
 1. Compute the error at the output layer: sigmoid_d(output) * (difference between expected and computed outputs)
    for each output
 2. Compute the error at the hidden layer: sigmoid_d(hidden) * 
	sum(weights_o_h * difference between expected output and computed output at output layer)
	for each hidden layer node
 3. Adjust the weights from the hidden to the output layer: learning rate * error at the output layer * error at the hidden layer
    for each connection between the hidden and output layers
 4. Adjust the weights from the input to the hidden layer: learning rate * error at the hidden layer * input layer node value
    for each connection between the input and hidden layers
 5. REMEMBER TO FREE ANY ALLOCATED MEMORY WHEN YOU'RE DONE (or use std::vector ;)
*/
void CNeuralNet::propagateErrorBackward(std::vector<double> desiredOutput)
{
	std::vector<double> _expOutputs = desiredOutput; //store the input in a reference vector

	double weightShift = 0;

	// Compute error at output layer and adjust the weights
	for (int i = 0; i < m_outputLayerSize; ++i)
	{
		// Calculate the error value
		double error = _outputActivation[i] * (1 - _outputActivation[i]) * (_expOutputs[i] - _outputActivation[i]);

		//Store the error value in the neuron  
		m_vecLayer[1].m_vecNeurons[i].m_Error = error;

		// Iterate through the neurons and adjust the weights based on the error and the learning rate
		for (int j = 0; j < m_vecLayer[1].m_vecNeurons[i].m_vecWeight.size(); ++j)
		{
			//calculate the shift that must happen in the weight
			weightShift = error * m_lRate * m_vecLayer[0].m_vecNeurons[j].m_Activation;

			//Shift the weight 
			m_vecLayer[1].m_vecNeurons[i].m_vecWeight[j] += weightShift;

			// Store the shift occurence 
			m_vecLayer[1].m_vecNeurons[i].m_prevWeight[j] = weightShift;
		}

	}

	//Now we move back to the hidden layer and calculate the error here
	for (int i = 0; i < m_vecLayer[0].m_vecNeurons.size(); ++i)
	{
		double error = 0; //store the error

		// Sum the error*weights in the output layer
		for (int j = 0; j < m_vecLayer[1].m_vecNeurons.size(); ++j)
		{
			error += m_vecLayer[1].m_vecNeurons[j].m_Error * m_vecLayer[1].m_vecNeurons[j].m_vecWeight[i];
		}

		// Calculate the error   
		error *= m_vecLayer[0].m_vecNeurons[i].m_Error * (1 - m_vecLayer[0].m_vecNeurons[i].m_Error);
		m_vecLayer[0].m_vecNeurons[i].m_Error = error;

		//recalculate weights based on the error value 
		for (int k = 0; k < m_inputLayerSize; ++k)
		{
			weightShift = error * m_lRate * _expOutputs[k];

			//shift the weight
			m_vecLayer[0].m_vecNeurons[i].m_vecWeight[k] += weightShift;
			//Store the shift occurence 
			m_vecLayer[0].m_vecNeurons[i].m_prevWeight[k] = weightShift;
		}


	}
	//std::cout << "Backprop complete" << std::endl;
}

//This computes the mean squared error
	double CNeuralNet::meanSquaredError(std::vector<double> desiredOutput)
	{
			float sum = 0;
			float error;

			for (int i = 0; i < m_outputLayerSize; ++i)
			{
				error = desiredOutput[i] - _outputActivation[i];
				sum += (error*error);
			}
			//std::cout << "MSE complete" << std::endl;
			return sum / m_outputLayerSize;
		
	}
/**
This trains the neural network according to the back propagation algorithm.
The primary steps are:
for each training pattern:
  feed forward
  propagate backward
until the MSE becomes suitably small
*/
	void CNeuralNet::train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs, uint trainingSetSize)
{

	while (m_MSE > m_mse_cutoff)
	{
		for (int n = 0; n < trainingSetSize; ++n)
		{

			// Feed forward and update each nodes outputs
			feedForward(inputs[n]);

			// Propagate backwards 
			propagateErrorBackward(outputs[n]);

			// Update the MSE
			m_MSE = meanSquaredError(outputs[n]);
			//cout << m_MSE << endl;
			// Once the cutoff value is met the network is considered trained 
			if (m_MSE < m_mse_cutoff)
			{
				break;
			}
		}
		
	}
	//std::cout << "Training complete" << std::endl;
}

/**
Once our network is trained we can simply feed it some input though the feed forward
method and take the maximum value as the classification
*/
	uint CNeuralNet::classify(std::vector<double> input)
	{
	feedForward(input);

	float largest = 0;
	float largestIndex = 0;

	for (int i = 0; i < _outputActivation.size(); ++i)
	{
		if (_outputActivation[i] > largest)
		{
			largest = _outputActivation[i];
			largestIndex = i;
		}
	}
	//std::cout << "Classifying complete" << std::endl;
	//cout << largestIndex << endl;
	return largestIndex;
}

/**
Gets the output at the specified index
*/
double CNeuralNet::getOutput(uint index) const
{
	//std::cout << "getOutput complete" << std::endl;
	return _outputActivation[index];
}