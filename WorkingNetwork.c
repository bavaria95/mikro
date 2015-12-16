#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define INPUT_NUM 784
#define HIDDEN_NUM 42
#define HID_WEIGHTS_NUM INPUT_NUM + 1
#define OUTPUT_NUM 10
#define OUT_WEIGHTS_NUM HIDDEN_NUM + 1


double sigmoidFun(double x){
    double result = 0.0;

    result = 1.0/(1.0 + exp(-x));

    return result;
}

int isZero(double number){
    if(number < 0.01)
        return 1;

    return 0;
}

int isOne(double number){
    if(number > 0.99)
        return 1;

    return 0;
}

double * neuralNetwork( double inputArray[] ){
    int i, j, img;
    char * config = "config73";
    FILE * file;

    double inputLayer[INPUT_NUM];

    for(i = 0; i < INPUT_NUM; i++)
        inputLayer[i] = inputArray[i];

    double hiddenLayer[HIDDEN_NUM];

    for(i = 0; i < HIDDEN_NUM; i++)
        hiddenLayer[i] = 0.0;


    double * outputLayer = malloc(OUTPUT_NUM * sizeof(double));

    for(i = 0; i < OUTPUT_NUM; i++)
            outputLayer[i] = 0.0;

    double hiddenLayerWeights[HIDDEN_NUM][HID_WEIGHTS_NUM];
    double outputLayerWeights[OUTPUT_NUM][OUT_WEIGHTS_NUM];


    file = fopen(config, "rb");

    if(file == NULL){
        perror("File cannot be opened");
        return -1;
    }

    for(i = 0; i < HIDDEN_NUM; i++ ){
        for(j = 0; j < HID_WEIGHTS_NUM; j++)
            fread(&hiddenLayerWeights[i][j], sizeof(double), 1, file);
    }

    for(i = 0; i < OUTPUT_NUM; i++ ){
        for(j = 0; j < OUT_WEIGHTS_NUM; j++){
            fread(&outputLayerWeights[i][j], sizeof(double), 1, file);
        }
    }

    fclose(file);

    double weightedSum = 0.0;
    double value = 0.0;

    for( i = 0; i < HIDDEN_NUM; i++){
        weightedSum = hiddenLayerWeights[i][HID_WEIGHTS_NUM - 1];

        for( j = 0; j < HID_WEIGHTS_NUM - 1; j++){
            weightedSum += inputLayer[j] * hiddenLayerWeights[i][j];
        }

        value = sigmoidFun(weightedSum);
        hiddenLayer[i] = value;
    }

    for(i = 0; i < OUTPUT_NUM; i++){
        weightedSum = outputLayerWeights[i][OUT_WEIGHTS_NUM - 1];

        for(j = 0; j < OUT_WEIGHTS_NUM - 1; j++){
            weightedSum += hiddenLayer[j] * outputLayerWeights[i][j];
        }

        outputLayer[i] = sigmoidFun(weightedSum);
    }

    return outputLayer;
}