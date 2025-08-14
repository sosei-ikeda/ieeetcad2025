#include "hls_math.h"
#include "win.h"

#define NU 12
#define NY 9
#define NS_MAX 370

#define NX 10

class DFR_class{
private:
    float A;
    float B;
    float beta;
    float J[NX];
    float temp;
    float X_0[NX];
    float X_1[NX];
    double R[NX*NX];
    double Wout[NY][NX*NX+1];
    float Y[NY];
    float y_max;
    float y;
    float total_loss;
    float best_loss;
    int num_data;
    int num_epoch;

    double dA;
    double dB;
    double dW[NY][NX*NX+1];
    double RRT[(NX*NX+1)*(NX*NX+2)/2];

public:
    float best_A;
    float best_B;
    int D[NY];

    void init(float params[3]);
    void update(volatile float *U);
    float output();
    void reset();
    void optim_res();
    void optim_out();
    void grad_init();
    void grad_fit();
    void grad_epoch();
    void tikhonov_init();
    void tikhonov_update();
    void tikhonov_inv();
};

void DFR_class::init(float params[3]){
    A = params[0];
    B = params[1];
    beta = params[2];
    for(int i=0; i<NX; i++){
        X_0[i] = 0;
    }
    for(int i=0; i<NX; i++){
        X_1[i] = 0;
    }
    for(int i=0; i<NX*NX; i++){
        R[i] = 0;
    }
    for(int i=0; i<(NX*NX+1)*(NX*NX+2)/2; i++){
        RRT[i] = 0;
    }
}

void DFR_class::update(volatile float *U){
    for(int i=0; i<NX; i++){
        J[i] = 0;
        for(int j=0; j<NU; j++){
            J[i] += Win[j][i]*U[j];
        }
    }

    temp = X_0[0]+J[0];
    X_1[0] = X_0[NX-1]*B+A*temp;
    for(int i=1; i<NX; i++){
        temp = X_0[i]+J[i];
        X_1[i] = X_1[i-1]*B+A*temp;
    }

    for(int i=0; i<NX; i++){
        for(int j=0; j<NX; j++){
            R[i*NX+j] += X_1[i]*X_0[j];
        }
    }

    for(int i=0; i<NX; i++){
        X_0[i] = X_1[i];
    }
}

float DFR_class::output(){
    y = 0;

    for(int i=0; i<NY; i++){
        Y[i] = 0;
        for(int j=0; j<NX*NX; j++){
            Y[i] += Wout[i][j]*R[j];
        }
        Y[i] += Wout[i][NX*NX];
    }

    y_max = Y[0];
    for(int i=1; i<NY; i++){
        if(Y[i]>y_max){
            y = i;
            y_max = Y[i];
        }
    }

    return y;
}

void DFR_class::reset(){
    for(int i=0; i<NX; i++){
        X_0[i] = 0;
    }
    for(int i=0; i<NX; i++){
        X_1[i] = 0;
    }
    for(int i=0; i<NX*NX; i++){
        R[i] = 0;
    }
}

void DFR_class::optim_res(){
    float lr = 1;
    if(int(num_epoch/5)==1){
        lr *= 1e-1;
    }
    else if(int(num_epoch/5)==2){
        lr *= 1e-2;
    }
    else if(int(num_epoch/5)==3){
        lr *= 1e-3;
    }
    else if(int(num_epoch/5)==4){
        lr *= 1e-4;
    }

    float A_new = A;
    float B_new = B;

    A_new -= lr*dA;
    B_new -= lr*dB;

    if(0 <= A_new and A_new < 0.5 and 0 <= B_new and B_new < 0.5){
        A = A_new;
        B = B_new;
    }
}

void DFR_class::optim_out(){
    float lr = 1;
    if(int(num_epoch/5)==2){
        lr *= 1e-1;
    }
    else if(int(num_epoch/5)==3){
        lr *= 1e-2;
    }
    else if(int(num_epoch/5)==4){
        lr *= 1e-3;
    }

    for(int i=0; i<NY; i++){
        for(int j=0; j<NX*NX; j++){
            Wout[i][j] -= lr*dW[i][j];
        }
        Wout[i][NX*NX] = lr*dW[i][NX*NX];
    }
}

void DFR_class::grad_init(){
    best_A = A;
    best_B = B;
    total_loss = 0;
    best_loss = 100;
    num_epoch = 0;
    for(int i=0; i<NY; i++){
        for(int j=0; j<NX*NX+1; j++){
            Wout[i][j] = 0;
        }
    }
}

void DFR_class::grad_fit(){
    double temp_exp;
    double exp_sum = 0;
    int d;
    double dy[NY];
    double dr[NX*NX];
    double dx[NX];

    output();
    for(int i=0; i<NY; i++){
        temp_exp = Y[i] - y_max;
        Y[i] = exp(temp_exp);
        exp_sum += Y[i];
        if(D[i] == 1){
            d = i;
        }
    }
    for(int i=0; i<NY; i++){
        Y[i] = Y[i]/exp_sum;
        dy[i] = Y[i] - D[i];
    }

//    total_loss += -log(Y[d]+1e-7);
//    total_loss = total_loss/(Y[d]+1e-7);
    total_loss += 1/(Y[d]+1e-7)-1;


// back propagate @ output layer
    for(int i=0; i<NX*NX; i++){
        dr[i] = 0;
        for(int j=0; j<NY; j++){
            dr[i] += Wout[j][i] * dy[j];
        }
    }

    for(int i=0; i<NY; i++){
        for(int j=0; j<NX*NX; j++){
            dW[i][j] = dy[i] * R[j];
        }
        dW[i][NX*NX] = dy[i];
    }

// back propagate @ DPRR layer
    for(int i=0; i<NX; i++){
        dx[i] = 0;
        for(int j=0; j<NX; j++){
            dx[i] += X_0[i] * dr[i*NX+j];
        }
    }

// back propagate @ reservoir layer
    dA = 0;
    dB = 0;
    for(int i =0; i<NX; i++){
        dA += (J[i]+X_0[i]) * dx[i];
        if(i != 0){
            dB += X_1[i-1] * dx[i];
        }
    }

// parameter update
    optim_res();
    optim_out();

    num_data += 1;
}

void DFR_class::grad_epoch(){
    float avg_loss = total_loss/num_data;
    std::cout << A << "," << B << "," <<  avg_loss << std::endl;
    if(best_loss > avg_loss){
        best_loss = avg_loss;
        best_A = A;
        best_B = B;
    }
    num_data = 0;
    total_loss = 0;
//    total_loss = 1;

    num_epoch += 1;
}

void DFR_class::tikhonov_init(){
    for(int i=0; i<NY; i++){
        for(int j=0; j<NX*NX+1; j++){
            Wout[i][j] = 0;
        }
    }
}

void DFR_class::tikhonov_update(){
    int counter = 0;
    for(int i=0; i<NX*NX; i++){
        for(int j=0; j<i+1; j++){
            RRT[counter] += R[i]*R[j];
            counter += 1;
        }
    }
    for(int i=0; i<NX*NX; i++){
        RRT[counter] += R[i];
        counter += 1;
    }
    RRT[counter] += 1;

    for(int i=0; i<NY; i++){
        for(int j=0; j<NX*NX; j++){
            Wout[i][j] += D[i]*R[j];
        }
        Wout[i][NX*NX] += D[i];
    }
}

void DFR_class::tikhonov_inv(){
    int n = NX*NX+1;
    double temp_inv;
    double temp_sub1;
    double temp_sub2;
    double temp_sub3;
    int counter_1;
    int counter_2;

    for(int i=0; i<n; i++){
        RRT[(i*i+i)/2+i] += beta;
    }

    for(int i=0; i<n; i++){
        counter_1 = (i*i+i)/2;
        temp_inv = RRT[counter_1+i];
        for(int j=0; j<i; j++){
            temp_inv -= RRT[counter_1+j]*RRT[counter_1+j];
        }
        RRT[counter_1+i] = sqrt(temp_inv);
        for(int j=i+1; j<n; j++){
            counter_2 = (j*j+j)/2;
            temp_sub1 = RRT[counter_2+i];
            for(int k=0; k<i; k++){
                temp_sub1 -= RRT[counter_1+k]*RRT[counter_2+k];
            }
            RRT[counter_2+i] = temp_sub1/RRT[counter_1+i];
        }
    }

    for(int i=0; i<NY; i++){
        for(int j=0; j<n; j++){
            counter_1 = (j*j+j)/2;
            temp_sub2 = Wout[i][j];
            for(int k=0; k<j; k++){
                temp_sub2 -= Wout[i][k]*RRT[counter_1+k];
            }
            Wout[i][j] = temp_sub2/RRT[counter_1+j];
        }
    }

    for(int i=0; i<NY; i++){
        for(int j=n-1; j>=0; j--){
            temp_sub3 = Wout[i][j];
            for(int k=n-1; k>j; k--){
                temp_sub3 -= Wout[i][k]*RRT[(k*k+k)/2+j];
            }
            Wout[i][j] = temp_sub3/RRT[(j*j+j)/2+j];
        }
    }
}
