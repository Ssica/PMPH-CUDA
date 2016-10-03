#include<stdio.h>
#include<stdlib.h>

//Task 1.a)

//sequential version of transpose
template<class T>
void task1a(T* in_, T* out_, int rows, int cols) {
	for (int i = 0; i < rows; i++) { 
		for (int j = 0; j < cols; j++) {
            out_[j*rows+i] = in_[i*cols+j];
		}
	}
}
