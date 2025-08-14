#include <iostream>
#include <fstream>
#include <string>
#include "sim.h"
#include "train_d.h"
#include "test_d.h"

void split_U(std::string str, char del, float *arr){
    int first = 0;
    int last = str.find_first_of(del);
    int i = 0;
    while (first < str.size()) {
        std::string subStr(str, first, last - first);
        arr[i] = ((float)stof(subStr));
        first = last + 1;
        last = str.find_first_of(del, first);
        if (last == std::string::npos) {
            last = str.size();
        }
        i++;
    }
}
void csv2arr_U(std::string filename, float *arr, int ignore_line_num = 0){
    std::ifstream reading_file;
    reading_file.open(filename, std::ios::in);
    if(!reading_file){
        std::cerr << filename << " is not found" << std::endl;
    }
    std::string reading_line_buffer;
    for(int line = 0; line < ignore_line_num; line++){
        getline(reading_file, reading_line_buffer);
        if(reading_file.eof()) break;
    }
    int i = 0;
    while(std::getline(reading_file, reading_line_buffer)){
        if(reading_line_buffer.size() == 0) break;
        split_U(reading_line_buffer, ',', &arr[i*(NU+1)]);
        i++;
    }
}

void DFR(volatile float *U, volatile int *D, volatile int *Y, volatile int *IDX,
         volatile float *P, volatile int *END, int Ns, int Nt, int Nu, int Ny, int mode);

int main(){
//        float params[3] = {0.06, 0.48, 1e-6};
    float params[3] = {0.01, 0.01, 1e-8};
    float correct;
    float train_U[TRAIN*T*(NU+1)];
    float test_U[TEST*T*(NU+1)];
    int train_Y[TRAIN];
    int test_Y[TRAIN];
    int end[1];

    for(int i=0; i<TRAIN; i++){
        csv2arr_U(dataset + "/TRAIN_"+std::to_string(i)+".csv", &train_U[i*T*(NU+1)]);
    }
    for(int i=0; i<TEST; i++) {
        csv2arr_U(dataset + "/TEST_" + std::to_string(i) + ".csv", &test_U[i*T*(NU+1)]);
    }

    end[0] = 0;
    DFR(train_U,train_D,train_Y,idx_arr,params,end,TRAIN,T,NU,NY,25);
    std::cout << params[0] << "," << params[1] << "," << params[2] << std::endl;

    end[0] = 0;
    DFR(train_U,train_D,train_Y,idx_arr,params,end,TRAIN,T,NU,NY,0);

    end[0] = 0;
    DFR(test_U,test_D,test_Y,idx_arr,params,end,TEST,T,NU,NY,1);

    correct = 0;
    for(int i=0; i<TEST; i++){
        if (test_Y[i] == test_D[i]) {
            correct += 1;
        }
    }
    std::cout << "TEST_ACC:" << (correct/TEST) << std::endl;

    return 0;
}