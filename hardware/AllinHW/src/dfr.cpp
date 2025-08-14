#include <string.h>
#include "dfr.h"

void DFR(volatile float *U, volatile int *D, volatile int *Y, volatile int *IDX,
         volatile float *P, volatile int *END, int Ns, int Nt, int Nu, int Ny, int mode) {
#pragma HLS INTERFACE m_axi port=U depth=50 offset=slave
#pragma HLS INTERFACE m_axi port=D depth=50 offset=slave
#pragma HLS INTERFACE m_axi port=Y depth=50 offset=slave
#pragma HLS INTERFACE m_axi port=IDX depth=50 offset=slave
#pragma HLS INTERFACE m_axi port=P depth=50 offset=slave
#pragma HLS INTERFACE m_axi port=END depth=50 offset=slave
#pragma HLS INTERFACE s_axilite port=Ns
#pragma HLS INTERFACE s_axilite port=Nt
#pragma HLS INTERFACE s_axilite port=Nu
#pragma HLS INTERFACE s_axilite port=Ny
#pragma HLS INTERFACE s_axilite port=mode
#pragma HLS INTERFACE s_axilite port=return
// mode => 0:gradient descent, 1:ridge regression, 2:prediction
    int Y_buff[NS_MAX];
    float P_buff[3];
    int END_buff[1];

    memcpy(P_buff, (const float*)P, 3*sizeof(float));
    memcpy(END_buff, (const int*)END, sizeof(int));

    static DFR_class dfr;

    dfr.init(P_buff);

    if(mode==0){
        dfr.tikhonov_init();
        for(int i=0; i<Ns; i++) {
            for (int j=0; j<Ny; j++) {
                dfr.D[j] = 0;
            }
            dfr.D[D[i]] = 1;
            for(int j=0; j<Nt; j++){
                dfr.update(&U[i*Nt*(Nu+1)+j*(Nu+1)]);
                if(U[i*Nt*(Nu+1)+j*(Nu+1)+Nu]!=0 or j==Nt-1){
                    dfr.tikhonov_update();
                    dfr.reset();
                    break;
                }
            }
        }
        dfr.tikhonov_inv();
    }
    else if(mode==1){
        for(int i=0; i<Ns; i++){
            for(int j=0; j<Nt; j++){
                dfr.update(&U[i*Nt*(Nu+1)+j*(Nu+1)]);
                if(U[i*Nt*(Nu+1)+j*(Nu+1)+Nu]!=0 or j==Nt-1){
                    Y_buff[i] = dfr.output();
                    dfr.reset();
                    break;
                }
            }
        }
    }
    else{
        dfr.grad_init();
        for(int epoch=0; epoch<mode; epoch++){
            for(int i=0; i<Ns; i++) {
                for (int j=0; j<Ny; j++) {
                    dfr.D[j] = 0;
                }
                dfr.D[D[IDX[epoch*Ns+i]]] = 1;
                for(int j=0; j<Nt; j++){
                    dfr.update(&U[IDX[epoch*Ns+i]*Nt*(Nu+1)+j*(Nu+1)]);
                    if(U[IDX[epoch*Ns+i]*Nt*(Nu+1)+j*(Nu+1)+Nu]!=0 or j==Nt-1){
                        dfr.grad_fit();
                        dfr.reset();
                        break;
                    }
                }
            }
            dfr.grad_epoch();
        }
        P_buff[0] = dfr.best_A;
        P_buff[1] = dfr.best_B;
    }

    END_buff[0] = 1;

    memcpy((int*)Y, Y_buff, Ns*sizeof(int));
    memcpy((float*)P, P_buff, 3*sizeof(float));
    memcpy((int*)END, END_buff, sizeof(int));
}