#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BINARY_STEP 0
#define SIGMOID 1

#define LAYERS 3
#define INPUT 2
#define HIDDEN 2
#define OUTPUT 1
#define EXAMPLES 4

typedef struct NN {
    int size[LAYERS+1]; // Input isn't counted as a layer as they aren't real neurons
    double ***Weights;
    double **Neurons;
    double **Deltas;
    double ***WeightDeltas;

    double Inputs[EXAMPLES][INPUT]; //bias
    double Ouputs[EXAMPLES][OUTPUT];
}NN;

double Binary_Step(double potential){
    if(potential>=0){
        return 1;
    }else{
        return 0;
    }
}

double Soft_Step(double potential){ //Sigmoid
    return (double) 1.0/(double)(1.0 + exp(-potential));
}

double Activation_Function(double potential, int type){ //Possible to add more activation function as needed
    if(type==0){
        return Binary_Step(potential);
    }else if(type==1){
        return Soft_Step(potential);
    }else{
        exit(1);
    }
}

void Initialize_Weights_Random(NN* ann){ //Possible to extend to many more dimensions
    srand(1);
    for(int i=0;i<INPUT+1;i++){
        for(int j=0;j<HIDDEN+1;j++){
            ann->Weights[0][i][j] = (double)rand()/(double)RAND_MAX*2.0-1.0;
        }
    }
    for(int i=1;i<LAYERS-1;i++){
        for(int j=0;j<HIDDEN;j++){
            for(int k=0;k<HIDDEN;k++){
                ann->Weights[i][j][k] = (double)rand()/(double)RAND_MAX*2.0-1.0;
            }
        }
    }
    for(int i=0;i<HIDDEN;i++){
        for(int j=0;j<OUTPUT;j++){
            ann->Weights[LAYERS-1][i][j] = (double)rand()/(double)RAND_MAX*2.0-1.0;
        }
    }
}

void Calculate_Layer(NN* ann, int layer){
    int sum = 0;
    if(layer == 1){ // Input Layer
        for(int j=0;j<HIDDEN;j++){
            for(int k=0;k<INPUT;k++){
                sum += ann->Neurons[0][k] * ann->Weights[0][k][j];
            }
            ann->Neurons[1][j] = Activation_Function(sum, SIGMOID);
            sum = 0;
        }
    }else if(layer == LAYERS){ // Output layer
        for(int j=0;j<OUTPUT;j++){
            for(int k=0;k<HIDDEN;k++){
                sum += ann->Neurons[LAYERS-1][k] * ann->Weights[LAYERS-1][k][j];
            }
            ann->Neurons[LAYERS][j] = Activation_Function(sum, SIGMOID);
            sum = 0;
        }
    }else{ // Hidden layer
        for(int j=0;j<HIDDEN;j++){
            for(int k=0;k<HIDDEN;k++){
                sum += ann->Neurons[layer-1][k] * ann->Weights[layer-1][k][j];
            }
            ann->Neurons[layer][j] = Activation_Function(sum, SIGMOID);
            sum = 0;
        }
    }
}

void Run_Network(NN* ann){
    for(int i=1;i<LAYERS+1;i++){
        Calculate_Layer(ann, i);
    }
}

void Calculate_Delta_Output(NN* ann, int nNeuron, double expected, double gamma){
    double result = ann->Neurons[LAYERS][nNeuron];
    ann->Deltas[LAYERS][nNeuron] = (double) gamma * result * (1 - result) * (expected - result);
}

double Calculate_Weight_Delta(double delta, double result, double learningRate){
    return learningRate * delta * result;
}

void Calculate_Delta_Hidden(NN* ann, int nNeuron, int layer, double gamma){
    double sum = 0;
    double result = ann->Neurons[layer][nNeuron];
    for(int i=0;i<ann->size[layer+1];i++){
        sum += ann->Deltas[layer+1][i] /*<--ERROR*/ * ann->Weights[layer][nNeuron][i];
    }
    ann->Deltas[layer][nNeuron] = gamma * result * (1 - result) * sum;
}

void Backpropagate_Output(NN* ann, double expected, double gamma, double learningRate){
    double weightDelta;
    for(int i=0;i<OUTPUT;i++){
        Calculate_Delta_Output(ann,i,expected,gamma);
        for(int x=0;x<HIDDEN;x++){
            weightDelta = Calculate_Weight_Delta(ann->Deltas[LAYERS][i], ann->Neurons[LAYERS-1][x],learningRate);
            ann->WeightDeltas[LAYERS-1][x][i] = weightDelta;
        }
    }
}

void Backpropagate_Hidden(NN* ann, int layer, double gamma, double learningRate){
    double weightDelta;
    for(int i=0;i<HIDDEN;i++){
        Calculate_Delta_Hidden(ann,i,layer,gamma);
        for(int x=0;x<HIDDEN;x++){
            weightDelta = Calculate_Weight_Delta(ann->Deltas[layer][i], ann->Neurons[layer-1][x],learningRate);
            ann->WeightDeltas[layer-1][x][i] = weightDelta;
        }
    }
}

void Learn_Weights(NN* ann){
    for(int i=0;i<INPUT+1;i++){
        for(int j=0;j<HIDDEN;j++){
            ann->Weights[0][i][j] += ann->WeightDeltas[0][i][j];
        }
    }
    for(int i=1;i<LAYERS-1;i++){
        for(int j=0;j<HIDDEN;j++){
            for(int k=0;k<HIDDEN;k++){
                ann->Weights[i][j][k] += ann->WeightDeltas[i][j][k];
            }
        }
    }
    for(int i=0;i<HIDDEN;i++){
        for(int j=0;j<OUTPUT;j++){
            ann->Weights[LAYERS-1][i][j] += ann->WeightDeltas[LAYERS-1][i][j];
        }
    }
}

void Train_Network(NN* ann, double input[][INPUT], double output[][OUTPUT], int inps, int epochs){
    for(int y=0;y<epochs;y++){
        for(int x=0;x<inps;x++){
            for(int i=0;i<ann->size[0]-1;i++){
                ann->Neurons[0][i] = input[x][i];
            }
            Run_Network(ann);
            Backpropagate_Output(ann,output[x][0],1,0.03);
            Backpropagate_Hidden(ann,2,1,0.03);
            Backpropagate_Hidden(ann,1,1,0.03);
            Learn_Weights(ann);
            //Report_Weights(ann,x);
        }
    }
    Run_Network(ann);
}

NN* Initialize_Network(int s_Input, int s_Hidden, int s_Output){
    NN* ann = malloc(sizeof(NN));
    if(ann){
        ann->Weights =(double***) malloc(LAYERS*sizeof(double**));
        ann->WeightDeltas =(double***) malloc(LAYERS*sizeof(double**));
        ann->Neurons =(double**) malloc((LAYERS+1)*sizeof(double*));
        ann->Deltas =(double**) malloc((LAYERS+1)*sizeof(double*));

        ann->Weights[0] =(double**) malloc((INPUT+1)*sizeof(double*));
        ann->WeightDeltas[0] =(double**) malloc((INPUT+1)*sizeof(double*));
        for(int j=0;j<INPUT+1;j++){
            ann->Weights[0][j] =(double*) malloc((HIDDEN+1)*sizeof(double));
            ann->WeightDeltas[0][j] =(double*) malloc((HIDDEN+1)*sizeof(double));
        }
        for(int i=1;i<LAYERS-1;i++){ //Weights[3][784][16]
             ann->Weights[i] =(double**) malloc((HIDDEN+1)*sizeof(double*));
             ann->WeightDeltas[i] =(double**) malloc((HIDDEN+1)*sizeof(double*));
             for(int j=0;j<HIDDEN;j++){
                ann->Weights[i][j] =(double*) malloc((HIDDEN+1)*sizeof(double));
                ann->WeightDeltas[i][j] =(double*) malloc((HIDDEN+1)*sizeof(double));
             }
        }
        ann->Weights[LAYERS-1] = (double**) malloc((HIDDEN+1)*sizeof(double*));
        ann->WeightDeltas[LAYERS-1] = (double**) malloc((HIDDEN+1)*sizeof(double*));
        for(int i=0;i<HIDDEN;i++){
            ann->Weights[LAYERS-1][i] = (double*) malloc((OUTPUT+1)*sizeof(double));
            ann->WeightDeltas[LAYERS-1][i] = (double*) malloc((OUTPUT+1)*sizeof(double));
        }

        ann->Neurons[0] =(double*) calloc(INPUT,sizeof(double));
        ann->Deltas[0] =(double*) calloc(INPUT,sizeof(double));
        for(int i=1;i<LAYERS;i++){
            ann->Neurons[i] =(double*) calloc(HIDDEN,sizeof(double));
            ann->Deltas[i] =(double*) calloc(HIDDEN,sizeof(double));
        }
        ann->Neurons[LAYERS] =(double*) calloc(OUTPUT,sizeof(double));
        ann->Deltas[LAYERS] =(double*) calloc(OUTPUT,sizeof(double));

        Initialize_Weights_Random(ann);

    }else{
        fprintf(stderr, "There was an error while initializing the NN.\n");
    }

   return ann;
}

void Delete_Network(NN* ann){
    for(int j=0;j<INPUT;j++){
        free(ann->Weights[0][j]);
        free(ann->WeightDeltas[0][j]);
    }
    free(ann->Weights[0]);
    free(ann->WeightDeltas[0]);
    for(int i=1;i<LAYERS-1;i++){ //Weights[3][784][16]
        for(int j=0;j<HIDDEN;j++){
            free(ann->Weights[i][j]);
            free(ann->WeightDeltas[i][j]);
        }
        free(ann->Weights[i]);
        free(ann->WeightDeltas[i]);
    }
    for(int i=0;i<HIDDEN;i++){
        free(ann->Weights[LAYERS][i]);
        free(ann->WeightDeltas[LAYERS][i]);
    }
    free(ann->Weights[LAYERS-1]);
    free(ann->WeightDeltas[LAYERS-1]);

    for(int i=1;i<LAYERS-1;i++){
        free(ann->Neurons[i]);
        free(ann->Deltas[i]);
    }
    free(ann->Neurons[0]);
    free(ann->Deltas[0]);
    free(ann->Neurons[LAYERS]);
    free(ann->Deltas[LAYERS]);
}

int main()
{
    /*-----HYPERPARAMETERS-----*/
    int s_Input = INPUT+1;
    int s_Hidden = HIDDEN+1;
    int s_Output = OUTPUT;
    /*------------------------*/

    double input[4][2] = {{0,1},{1,1},{1,0},{0,0}};
    double output[4][1] = {{1},{0},{1},{0}};


    NN* ann = Initialize_Network(s_Input, s_Hidden, s_Output);

    ann->size[0] = s_Input;
    ann->size[1] = s_Hidden;
    ann->size[2] = s_Hidden;
    ann->size[3] = s_Output;

    Train_Network(ann, input, output, 4, 1);


    Delete_Network(ann);

    return 0;
}
