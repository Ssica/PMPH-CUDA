#include<stdio.h>
#include<stdlib.h>

//Task 3.a)

//sequential version of matmult on flat array
float* task3a(float* mat1, float* mat2, int row1, int col1, int col2) {
    float* mat_out = (float*) malloc(row1*col2);
    
    for(int i = 0; i < row1; i++) {
        for(int j = 0; j < col2; j++) {
            float res = 0;
            for(int k = 0; k < col1; k++) {
                res = res + mat1[i*col1 + k] * mat2[k*col2 + j];
            }
            mat_out[i * col2 + j] = res;
        }
    }
    return mat_out;
}