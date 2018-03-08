#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define BINARY_STEP 0
#define SIGMOID 1

#define LAYERS 3

#define INPUT 784
#define HIDDEN 15
#define OUTPUT 10

#define BIGGEST 784

#define EXAMPLES 2000

typedef struct NN {
    int size[LAYERS+1]; // Input isn't counted as a layer as they aren't real neurons

    double Weights[LAYERS][BIGGEST+1][BIGGEST+1];
    double WeightDeltas[LAYERS][BIGGEST+1][BIGGEST+1];
    double Neurons[LAYERS+1][BIGGEST+1];
    double Deltas[LAYERS+1][BIGGEST+1];

    double Inputs[EXAMPLES][INPUT]; //bias
    double Outputs[EXAMPLES][OUTPUT];
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

void Initialize_Biases(NN* ann){ /**!!!HARDCODED!!!**/
    ann->Neurons[0][INPUT] = 1.0;
    ann->Neurons[1][HIDDEN] = 1.0;
    ann->Neurons[2][HIDDEN] = 1.0;
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

NN* Initialize_Network(int s_Input, int s_Hidden, int s_Output){
    NN* ann = calloc(1,sizeof(NN));
    if(ann){
        ann->size[0]=s_Input;               /**!!!!HARDCODED!!!!**/
        ann->size[1]=s_Hidden;
        ann->size[2]=s_Output;

        Initialize_Weights_Random(ann);

        Initialize_Biases(ann);

        return ann;
    }else{
        fprintf(stderr,"Error.\n");
        return NULL;
    }
}

void Delete_Network(NN* ann){
    free(ann);
}

void Calculate_Layer(NN* ann, int layer){
    double sum = 0;
    if(layer == 1){ // Input Layer -> First hidden
        for(int j=0;j<HIDDEN;j++){
            for(int k=0;k<INPUT+1;k++){
                sum += ann->Neurons[0][k] * ann->Weights[0][k][j];
            }
            ann->Neurons[1][j] = Activation_Function(sum, SIGMOID);
            sum = 0;
        }
    }else if(layer == LAYERS){ // Last hidden -> Output layer
        for(int j=0;j<OUTPUT;j++){
            for(int k=0;k<HIDDEN+1;k++){
                sum += ann->Neurons[LAYERS-1][k] * ann->Weights[LAYERS-1][k][j];
            }
            ann->Neurons[LAYERS][j] = Activation_Function(sum, SIGMOID);
            sum = 0;
        }
    }else{ // Hidden layer -> Hidden layer
        for(int j=0;j<HIDDEN;j++){
            for(int k=0;k<HIDDEN+1;k++){
                sum += ann->Neurons[layer-1][k] * ann->Weights[layer-1][k][j];
            }
            ann->Neurons[layer][j] = Activation_Function(sum, SIGMOID);
            sum = 0;
        }
    }
}

void Run_Network(NN* ann){
    #pragma omp parallel for
    for(int i=1;i<LAYERS+1;i++){
        Calculate_Layer(ann, i);
    }
}

void Calculate_Delta_Output(NN* ann, int nNeuron, double expected, int gamma){
    double result = ann->Neurons[LAYERS][nNeuron];
    ann->Deltas[LAYERS][nNeuron] = gamma * result * (1 - result) * (expected - result);
}

double Calculate_WeightDelta(double result, double delta, double learningRate){
    return learningRate * delta * result;
}

void Calculate_Delta_HiddenLast(NN* ann, int nNeuron, int gamma){
    double sum = 0;
    double result = ann->Neurons[LAYERS-1][nNeuron];
    for(int i=0;i<OUTPUT;i++){
        sum += ann->Deltas[LAYERS][i] * ann->Weights[LAYERS-1][nNeuron][i];
    }
    ann->Deltas[LAYERS-1][nNeuron] = gamma * result * (1 - result) * sum;
}

void Calculate_Delta_Hidden(NN* ann, int layer, int nNeuron, int gamma){
    double sum = 0;
    double result = ann->Neurons[layer][nNeuron];
    for(int i=0;i<HIDDEN;i++){
        sum += ann->Deltas[layer+1][i] * ann->Weights[layer][nNeuron][i];
    }
    ann->Deltas[layer][nNeuron] = gamma * result * (1 - result) * sum;
}

void Backpropagate_Output(NN* ann, double expected[], double gamma, double learningRate){
    for(int i=0;i<OUTPUT;i++){
        Calculate_Delta_Output(ann, i, expected[i], gamma);
        for(int j=0;j<HIDDEN;j++){
            ann->WeightDeltas[LAYERS-1][j][i] = Calculate_WeightDelta(ann->Neurons[LAYERS-1][j], ann->Deltas[LAYERS][i], learningRate);
        }
    }
}

void Backpropagate_Hidden(NN* ann, int layer, double gamma, double learningRate){
    for(int i=0;i<HIDDEN;i++){
        Calculate_Delta_Hidden(ann, layer, i, gamma);
        for(int j=0;j<HIDDEN;j++){
            ann->WeightDeltas[layer-1][j][i] = Calculate_WeightDelta(ann->Neurons[layer-1][j], ann->Deltas[layer][i], learningRate);
        }
    }
}

void Backpropagate_Input(NN* ann, double gamma, double learningRate){
    for(int i=0;i<HIDDEN;i++){
        Calculate_Delta_Hidden(ann, 1, i, gamma);
        for(int j=0;j<INPUT;j++){
            ann->WeightDeltas[0][j][i] = Calculate_WeightDelta(ann->Neurons[0][j], ann->Deltas[1][i], learningRate);
        }
    }
}

void Learn_Weights(NN* ann){    // double Weights[LAYERS][BIGGEST+1][BIGGEST+1]; double WeightDeltas[LAYERS][BIGGEST+1][BIGGEST+1];
    for(int i=0;i<LAYERS;i++){
        for(int j=0;j<BIGGEST+1;j++){
            for(int k=0;k<BIGGEST+1;k++){
                ann->Weights[i][j][k] += ann->WeightDeltas[i][j][k];
            }
        }
    }
}

void Train_Network(NN* ann, double input[][INPUT], double output[][OUTPUT], int inps, int epochs){
    #pragma omp parallel for
    for(int epoch=0;epoch<epochs;epoch++){
        for(int inp=0;inp<inps;inp++){
            for(int i=0;i<INPUT;i++){
                ann->Neurons[0][i] = input[inp][i];
            }
            Run_Network(ann);
            Backpropagate_Output(ann, output[inp], 1.0, 0.08);  /**!!!HARDCODED!!!**/
            Backpropagate_Hidden(ann, 2, 1.0, 0.08);
            Backpropagate_Input(ann, 1.0, 0.08);
            Learn_Weights(ann);
        }
    }
}

void Read_Data_IDX(FILE* f, FILE* l, double input[EXAMPLES][INPUT], double output[EXAMPLES][OUTPUT]){
    fseek(f, 4, SEEK_SET);
    fseek(l, 4, SEEK_SET);

    unsigned char label;
    unsigned char tmp;
    for(int i=0;i<EXAMPLES;i++){
        for(int m=0;m<INPUT;m++){
            fread(&tmp,1,1,f);
            input[i][m] =(double) tmp;
        }
        fread(&label,1,1,l);
        for(int y=0;y<OUTPUT;y++){
            if(y==label){
                output[i][y] =(double) 1;
            }else{
                output[i][y] =(double) 0;
            }
        }
    }
}

int main()
{
    /*-----HYPERPARAMETERS-----*/
    int s_Input = INPUT+1;
    int s_Hidden = HIDDEN+1;
    int s_Output = OUTPUT;
    /*------------------------*/

    FILE *f = fopen("./data/images.idx", "r");
    FILE *l = fopen("./data/labels.idx", "r");

    /*double input[16][4] = {{0,0,0,0},{0,0,0,1},{0,0,1,1},{0,1,1,1},{1,1,1,1},{1,0,0,0},{1,1,0,0},{1,1,1,0},{1,0,0,1},{1,0,1,0},{1,0,1,1},{0,1,0,1},{0,1,1,0},{0,1,0,0},{0,0,1,0},{0,1,0,1}};
    double output[16][4] = {{0,0,0,0},{1,0,0,0},{1,1,0,0},{1,1,1,0},{1,1,1,1},{0,0,0,1},{0,0,1,1},{0,1,1,1},{1,0,0,1},{1,0,1,0},{1,1,0,1},{1,0,1,0},{0,1,1,0},{0,0,1,0},{0,1,0,0},{0,0,1,1}};*/

    NN* ann = Initialize_Network(s_Input, s_Hidden, s_Output);

    Read_Data_IDX(f, l, ann->Inputs, ann->Outputs);

    Train_Network(ann, ann->Inputs, ann->Outputs, EXAMPLES, 10);
    /*for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            for(int k=0;k<2;k++){
                for(int l=0;l<2;l++){
                    ann->Neurons[0][0] = i;
                    ann->Neurons[0][1] = j;
                    ann->Neurons[0][2] = k;
                    ann->Neurons[0][3] = l;
                    Run_Network(ann);
                    printf("%d - %d - %d - %d -> %f - %f - %f - %f\n",i,j,k,l,ann->Neurons[LAYERS][0], ann->Neurons[LAYERS][1], ann->Neurons[LAYERS][2], ann->Neurons[LAYERS][3]);
                }
            }
        }
    }*/

    for(int i=0;i<10;i++){
        for(int j=0;j<784;j++){
            ann->Neurons[0][j] = ann->Inputs[i][j];
        }
        Run_Network(ann);
        printf("\n%f %f %f %f %f %f %f %f %f %f <---> %f %f %f %f %f %f %f %f %f %f\n", ann->Outputs[i][0], ann->Outputs[i][1], ann->Outputs[i][2], ann->Outputs[i][3], ann->Outputs[i][4],
                    ann->Outputs[i][5], ann->Outputs[i][6], ann->Outputs[i][7], ann->Outputs[i][8], ann->Outputs[i][9], ann->Neurons[LAYERS][0], ann->Neurons[LAYERS][1], ann->Neurons[LAYERS][2],
                     ann->Neurons[LAYERS][3], ann->Neurons[LAYERS][4], ann->Neurons[LAYERS][5], ann->Neurons[LAYERS][6], ann->Neurons[LAYERS][7], ann->Neurons[LAYERS][8],
                      ann->Neurons[LAYERS][9]);
    }

    Run_Network(ann);

    printf("Out: %f\n",ann->Neurons[LAYERS][0]);

    Delete_Network(ann);

    fclose(f);
    fclose(l);
    return 0;
}
