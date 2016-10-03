#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<time.h>


//Taken from a former assignment
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1){
	unsigned int resolution=1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return (diff<0);
}

//used to validate the test run
void validate(float* mat1, float* mat2, int rows, int cols) {
	bool valid = true;
	for (int i = 0; i < rows; i++) {
		if (!valid) { break; }
		for (int j = 0; j < cols; j++) {
            
            printf("print mat1: %f", mat1[i*cols+j]);
            printf("print mat2: %f", mat2[j*cols+i]);
			//if (mat1[i*cols+j] == mat2[i*cols+j]) {
                
                //continue;
			//}
            //else{
            //    valid = false;
            //    printf("Failts at: row: %d, col: %d, ",i,j);
            //    break;
            //}
		}
	}
	
	if (valid) { printf("\n Result from test valid\n"); }
	else { printf("\n Result from test invalid\n"); }
}

//initialize matrix
void init_mat(float* mat,int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)(rand()%100+1);
    }
}
