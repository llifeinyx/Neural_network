#define MOORE 8
#define SIDE 4

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "NNStruct.h"

void GenerateEnvironment();

float environment[MOORE];
int rightAnswer[SIDE];


int main() {
    srand(time(0));
    NNetwork neuralNetwork;
    NNetworkConstruct(&neuralNetwork);
    //teaching extern-cycle
    for (int j = 0; j < 500; ++j) {
        //teaching endo-cycle
        for (int i = 0; i < 150; ++i) {
            GenerateEnvironment();
            SetInputs(&neuralNetwork, environment);
            NeuronCalc(&neuralNetwork);
            NeuronCalcError(&neuralNetwork, rightAnswer);
            WeightsUpdate(&neuralNetwork);
            GetAnswer(&neuralNetwork, rightAnswer);
        }
        printf("\n");
    }
    NNetworkDestruct(&neuralNetwork);
    return 0;
}

void GenerateEnvironment(){
    float maxSide = 0;
    float tempSide[SIDE];
    int sideIndex;
    for (int i = 0; i < MOORE; ++i) {
        environment[i] = (float)(rand()%101)/100;
        //printf("%.2f ", environment[i]);
    }
    for (int i = 0, j = 0; i < SIDE; ++i, j += 2) {
        //clear rightAnswer!
        rightAnswer[i] = 0;

        //count right index from tempSide
        if (i != SIDE - 1){
            tempSide[i] =  environment[j] + environment[j + 1] + environment[j + 2];
        } else {
            tempSide[i] = environment[j] + environment[j + 1] + environment[0];
        }
        if (tempSide[i] > maxSide){
            maxSide = tempSide[i];
            sideIndex = i;
        }
    }
    rightAnswer[sideIndex] = 1;
    //printf("\n%d", sideIndex );

}