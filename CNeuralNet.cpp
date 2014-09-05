/*
                                                                           
   (               )                                        )              
 ( )\     )     ( /(       (                  (  (     ) ( /((             
 )((_) ( /(  (  )\())`  )  )(   (  `  )   (   )\))( ( /( )\())\  (   (     
((_)_  )(_)) )\((_)\ /(/( (()\  )\ /(/(   )\ ((_))\ )(_)|_))((_) )\  )\ )  
 | _ )((_)_ ((_) |(_|(_)_\ ((_)((_|(_)_\ ((_) (()(_|(_)_| |_ (_)((_)_(_/(  
 | _ \/ _` / _|| / /| '_ \) '_/ _ \ '_ \/ _` |/ _` |/ _` |  _|| / _ \ ' \)) 
 |___/\__,_\__||_\_\| .__/|_| \___/ .__/\__,_|\__, |\__,_|\__||_\___/_||_|  
                    |_|           |_|         |___/                         

 For more information on back-propagation refer to:
 Chapter 18 of Russel and Norvig (2010).
 Artificial Intelligence - A Modern Approach.
 */

#include "CNeuralNet.h"
#include <iostream>

//Struct for a layer
SNeuronLayer::SNeuronLayer(int NumNeurons,
	int NumInputsPerNeuron) : m_iNumNeurons(NumNeurons)
{
	for (int i = 0; i<NumNeurons; ++i)

		m_vecNeurons.push_back(SNeuron(NumInputsPerNeuron));
}

//Struct for a Neuron
SNeuron::SNeuron(int NumInputs) : m_iNumInputs(NumInputs + 1),
m_dActivation(0),
m_dError(0)

{
	//we need an additional weight for the bias hence the +1   
	for (int i = 0; i<NumInputs + 1; ++i)
	{
		//set up the weights with an initial random value   
		m_vecWeight.push_back(0);	//initialized to 0 since they are overwritten later
	}
}


/**
 The constructor of the neural network. This constructor will allocate memory
 for the weights of both input->hidden and hidden->output layers, as well as the input, hidden
 and output layers.
*/
CNeuralNet::CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff) 
: m_inputLayerSize(inputLayerSize), m_hiddenLayerSize(hiddenLayerSize), m_outputLayerSize(outputLayerSize), m_lRate(lRate), m_mse_cutoff(mse_cutoff) //intializer list
{
	std::vector<double> _inputs(m_inputLayerSize);
	SNeuronLayer hiddenLayer(m_hiddenLayerSize, m_inputLayerSize);
	SNeuronLayer outputLayer(m_outputLayerSize, m_hiddenLayerSize);

	m_vecLayer.push_back(hiddenLayer);
	m_vecLayer.push_back(outputLayer);

	initWeights();
}

/**
 The destructor of the class. All allocated memory will be released here
*/
CNeuralNet::~CNeuralNet() //No pointers to be released just yet
{
	//TODO
}
/**

 Method to initialize the both layers of weights to random numbers
*/
void CNeuralNet::initWeights()
{
	for (int i = 0; i < 2; ++i) //iterate through the two layers (hidden and output layers)
	{
		for (int j = 0; j < m_vecLayer[i].m_iNumNeurons; ++j) //iterate through the neurons in each layer
		{
			for (int k = 0; k < m_vecLayer[i].m_vecNeurons[j].m_iNumInputs; ++k) //iterate through each input weight going into each neuron
			{
				//float randWeight = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2))); //generate a random float between -1 and 1
				float randWeight = RandomClamped();
				m_vecLayer[i].m_vecNeurons[j].m_vecWeight[k] = randWeight;	//assign the weight
			}
		}
	}
}

double CNeuralNet::Sigmoid(double netinput)
{
	return netinput / (1 + abs(netinput));
}


/**
 This is the forward feeding part of back propagation.
 1. This should take the input and copy the memory (use memcpy / std::copy)
 to the allocated _input array.
 2. Compute the output of at the hidden layer nodes 
 (each _hidden layer node = sigmoid (sum( _weights_h_i * _inputs)) //assume the network is completely connected
 3. Repeat step 2, but this time compute the output at the output layer
*/
void CNeuralNet::feedForward(const double * const inputs) 
{
	memcpy(&_inputs, inputs, sizeof(inputs)); //god I hope this works
	
	int cWeight = 0; 

	vector<double> outputs;

	//For each layer...   
	for (int i = 0; i < 2; ++i)
	{

		if (i > 0)
		{
			_inputs = outputs;
		}

		outputs.clear();

	//for each neuron sum the (inputs * corresponding weights).Throw    
	//the total at our sigmoid function to get the output.   
		for (int n = 0; n < m_vecLayer[i].m_iNumNeurons; ++n)
		{
			double netinput = 0;

			int NumInputs = m_vecLayer[i].m_vecNeurons[n].m_iNumInputs;

			//for each weight   
			for (int k = 0; k < NumInputs; ++k)
			{
				//sum the weights x inputs   
				netinput += m_vecLayer[i].m_vecNeurons[n].m_vecWeight[k] * _inputs[cWeight++];
			}

			//The combined activation is first filtered through the sigmoid    
			//function and a record is kept for each neuron    
			m_vecLayer[i].m_vecNeurons[n].m_dActivation = Sigmoid(netinput);

			//store the outputs from each layer as we generate them.   
			outputs.push_back(m_vecLayer[i].m_vecNeurons[n].m_dActivation);

			cWeight = 0;
		}
	}
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
void CNeuralNet::propagateErrorBackward(const double * const desiredOutput)
{
	//TODO
}
/**
This computes the mean squared error
A very handy formula to test numeric output with. You may want to commit this one to memory
*/
double CNeuralNet::meanSquaredError(const double * const desiredOutput){
	/*TODO:
	sum <- 0
	for i in 0...outputLayerSize -1 do
		err <- desiredoutput[i] - actualoutput[i]
		sum <- sum + err*err
	return sum / outputLayerSize
	*/
	return 1;
}
/**
This trains the neural network according to the back propagation algorithm.
The primary steps are:
for each training pattern:
  feed forward
  propagate backward
until the MSE becomes suitably small
*/
void CNeuralNet::train(const double** const inputs,
		const double** const outputs, uint trainingSetSize) {
	//TODO
}
/**
Once our network is trained we can simply feed it some input though the feed forward
method and take the maximum value as the classification
*/
uint CNeuralNet::classify(const double * const input){
	return 1; //TODO: fix me
}
/**
Gets the output at the specified index
*/
double CNeuralNet::getOutput(uint index) const{
	return 0; //TODO: fix me
}