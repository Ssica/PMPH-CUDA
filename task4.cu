#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>
const int listLength = 700;
__global__ void squareKernel(float* d_in, float *d_out, int threads_num) {
const unsigned int lid = threadIdx.x; // local id inside a block
const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
if (gid < threads_num){
	d_out[gid] = powf((d_in[gid]/(d_in[gid]-2.3)),3);
	}// do computation
}
int timeval_subtract(struct timeval* result,struct timeval* t2,struct timeval* t1) {
	unsigned int resolution=1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec) -(t1->tv_usec + resolution * t1->tv_sec) ;
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return (diff<0);
}

int main(int argc, char** arigv) {
	unsigned int num_threads = 75311;
	unsigned int mem_size = num_threads*sizeof(float);
	unsigned int block_size = 256;
	unsigned int num_blocks = ((num_threads + (block_size-1)) / block_size);
	unsigned long int elapsed1;
	unsigned long int elapsed2;
	struct timeval t_start, t_end, t_diff;
	float* h_in = (float*)malloc(mem_size);
	float* h_out = (float*)malloc(mem_size);
	float tmpList[listLength];
	float epsilon = 1*1e-4;
	for(unsigned int i = 0; i<num_threads; ++i){
		h_in[i] = (float)i;
	}
	
	
	//Serial mapping
	gettimeofday(&t_start, NULL);
	for(int i = 0; i < listLength; ++i){
		tmpList[i] = powf((h_in[i]/(h_in[i]-2.3)),3.0);
	}
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed1 = t_diff.tv_sec*1e6+t_diff.tv_usec;
	printf("Serial Mapping took %d microseconds (%.2fms)\n",elapsed1,elapsed1/1000.0);

    //Parallel Mapping
	float* d_in;
	float* d_out;
	cudaMalloc((void**)&d_in, mem_size);
	cudaMalloc((void**)&d_out, mem_size);

	cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
	gettimeofday(&t_start, NULL);
	squareKernel<<< num_blocks, block_size>>>(d_in, d_out, num_threads);
	cudaThreadSynchronize();
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed2 = t_diff.tv_sec*1e6+t_diff.tv_usec;
	printf("Parallel mapping took %d microseconds (%.2fms)\n",elapsed2,elapsed2/1000.0);

	cudaMemcpy(h_out, d_out, sizeof(float)*num_threads, cudaMemcpyDeviceToHost);

	unsigned int mep = 1;

	for(unsigned int i=0; i<listLength; ++i){
		if(abs(h_out[i] - tmpList[i]) > epsilon){
			mep = 0;
		}
	}
	if(mep == 1){
		std::cout<<"Valid\n";
	}
	if(mep == 0){
		std::cout<<"Invalid\n";
	}
	// clean-up memory
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);

}


