/*
 * CNeuralNet.h
 *
 *  Created on: 26 Dec 2013
 *      Author: benjamin
 */

#ifndef CNEURALNET_H_
#define CNEURALNET_H_
#include <vector>
#include <cmath>
#include "utils.h"
#include <algorithm>
#include <stdlib.h>
#include <cstring>
#include <stdio.h>
#include <stdint.h>

//Since different neurons take a different number of inputs, a struct to contain neurons was deemed to be best
struct SNeuron
{
	//the number of inputs into each neuron 
	int	m_iNumInputs;

	//the weights for each input 
	std::vector<double>	m_vecWeight;

	//The most recent weight update
	std::vector<double> m_prevWeight;

	//the activation of this neuron 
	double	m_Activation;

	//the error value 
	double	m_Error;

	//constructor
	SNeuron(int NumInputs);
};

struct SNeuronLayer
{
	//the number of neurons in this layer 
	int m_iNumNeurons;

	//the layer of neurons 
	std::vector<SNeuron>	m_vecNeurons;

	//constructor
	SNeuronLayer(int NumNeurons, int NumInputsPerNeuron);
};

typedef unsigned int uint;
class CBasicEA; //forward declare EA class, which will have the power to access weight vectors

class CNeuralNet {
	friend class CBasicEA;
protected:
	void feedForward(const double * const inputs); //you may modify this to do std::vector<double> if you want
	void propagateErrorBackward(const double * const desiredOutput); //you may modify this to do std::vector<double> if you want
	double meanSquaredError(const double * const desiredOutput); //you may modify this to do std::vector<double> if you want
public:
	//variables
	uint m_inputLayerSize;
	uint m_hiddenLayerSize;
	uint m_outputLayerSize; 
	double m_lRate;	//learning rate for back propogation
	double m_mse_cutoff;

	//vectors for the network
	std::vector<SNeuronLayer> m_vecLayer;
	std::vector<double> _inputs;
	std::vector<double> _expOutputs;
	std::vector<double> _outputActivation;

	//methods
	double Sigmoid(double netinput);
	CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff);
	void initWeights();
	void train(const double ** const inputs,const double ** const outputs, uint trainingSetSize); //you may modify this to do std::vector<std::vector<double> > or do boost multiarray or something else if you want
	uint classify(const double * const input); //you may modify this to do std::vector<double> if you want
	double getOutput(uint index) const;
	virtual ~CNeuralNet();
};

#endif /* CNEURALNET_H_ */
