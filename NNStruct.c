#include "NNStruct.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// set amount neurons on layer
const int INPUT_0 = 8, HIDDEN_1 = 8, HIDDEN_2 = 6, OUTPUT_3 = 4;

float Sigmoida(float x){
    return 1/(1 + powf(2.71828f,-x));
}
float SigmoidaDerivative(float x){
    return x * (1 - x);
}


void NNetworkConstruct(NNetwork *this){
    //set layers
    this->layers[0] = INPUT_0; this->layers[1] = HIDDEN_1; this->layers[2] = HIDDEN_2; this->layers[3] = OUTPUT_3;

    //set weights
    this->weights = (float***) malloc(LAYERS_AMOUNT * sizeof (float**));
    for (int i = 0; i < LAYERS_AMOUNT - 1; i++){
        this->weights[i] = (float**) malloc(this->layers[i] * sizeof (float*));
        for (int j = 0; j < this->layers[i]; ++j) {
            this->weights[i][j] = (float*) malloc(this->layers[i + 1] * sizeof(float));
            for (int k = 0; k < this->layers[i + 1]; ++k) {
                this->weights[i][j][k] = (float)((rand() % 101) - 50)/100;
                //printf("%.2f ", this.weights[i][j][k]);
            }
            //printf("\n");
        }
        //printf("\n\n\n");
    }

    //set neurons
    this->neuron = (Neuron**) malloc(LAYERS_AMOUNT * sizeof(Neuron*));
    for (int i = 0; i < LAYERS_AMOUNT; ++i) {
        this->neuron[i] = (Neuron*) malloc(this->layers[i] * sizeof (Neuron));
    }

    //end construct process
}

void NNetworkDestruct(NNetwork *this){
    //delete weights
    for (int i = 0; i < LAYERS_AMOUNT - 1; ++i) {
        for (int j = 0; j < this->layers[i]; ++j) {
            free(this->weights[i][j]);
        }
        free(this->weights[i]);
    }
    free(this->weights);

    //delete neurons
    for (int i = 0; i < LAYERS_AMOUNT; ++i) {
        free(this->neuron[i]);
    }
    free(this->neuron);
}

void SetInputs(NNetwork *this, const float *input){
    //set value for input layer
    for (int i = 0; i < INPUT_0; ++i) {
        this->neuron[0][i].value = input[i];
    }
}

void NeuronCalc(NNetwork *this){
    //Count value for all neurons
    for (int i = 1; i < LAYERS_AMOUNT; ++i) {
        for (int j = 0; j < this->layers[i]; ++j) {
            float tempSum = 0;
            for (int k = 0; k < this->layers[i - 1]; ++k) {
                tempSum += this->neuron[i - 1][k].value * this->weights[i - 1][k][j];
            }
            this->neuron[i][j].value = Sigmoida(tempSum);
        }
    }
}

void NeuronCalcError(NNetwork *this, const int *output){
    //count error output layer neurons
    for (int i = 0; i < OUTPUT_3; ++i) {
        this->neuron[3][i].error =
                ((float)output[i] - this->neuron[3][i].value) * SigmoidaDerivative(this->neuron[3][i].value);
    }

    //backpropagation
    for (int i = LAYERS_AMOUNT - 2; i > 0; --i) {
        for (int j = 0; j < this->layers[i]; ++j) {
            float tempSum = 0;
            for (int k = 0; k < this->layers[i + 1]; ++k) {
                tempSum += this->neuron[i + 1][k].error * this->weights[i][j][k];
            }
            this->neuron[i][j].error = tempSum * SigmoidaDerivative(this->neuron[i][j].value);
        }
    }
}

void WeightsUpdate(NNetwork *this){
    //weights update
    for (int i = 0; i < LAYERS_AMOUNT - 1; ++i) {
        for (int j = 0; j < this->layers[i]; ++j) {
            for (int k = 0; k < this->layers[i + 1]; ++k) {
                this->weights[i][j][k] += 0.1f * this->neuron[i][j].value * this->neuron[i + 1][k].error;
            }
        }
    }
}

void GetAnswer(NNetwork *this, const int *rAnswer){
    // get value max output
    float maxValue = 0;
    int sideIndex;
    for (int i = 0; i < OUTPUT_3; ++i) {
        if (this->neuron[3][i].value > maxValue){
            maxValue = this->neuron[3][i].value;
            sideIndex = i;
        }
    }
    if (rAnswer[sideIndex] == 1){
        printf("1");
    } else {
        printf("0");
    }
}

