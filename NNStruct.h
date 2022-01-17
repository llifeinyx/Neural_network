
#ifndef NEURAL_NETWORK_NNSTRUCT_H
#define NEURAL_NETWORK_NNSTRUCT_H

#define LAYERS_AMOUNT 4

typedef struct Neuron{
    float value;
    float error;
} Neuron;


typedef struct NNetwork{
    float*** weights;
    int layers[LAYERS_AMOUNT];
    Neuron **neuron;

} NNetwork;

float Sigmoida(float x);
float SigmoidaDerivative(float x);

void NNetworkConstruct(NNetwork *this);
void NNetworkDestruct(NNetwork *this);
void SetInputs(NNetwork *this, const float *input);
void NeuronCalc(NNetwork *this);
void NeuronCalcError(NNetwork *this, const int *output);
void WeightsUpdate(NNetwork *this);
void GetAnswer(NNetwork *this, const int *rAnswer);


#endif //NEURAL_NETWORK_NNSTRUCT_H
