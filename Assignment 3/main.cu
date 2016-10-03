#include "help.h"
#include "task1a.h"
#include "task1c.cu.h"
#include "task1d.cu.h"
#include "task2d.cu.h"
#define ROWS 1024
#define COLS 1024
#define TILE 32

template<class T, int tile>
void task1c_trans(float* in_, float* out_, int rows, int cols) {
	int dimx = ceil((float)cols/tile);
	int dimy = ceil((float)rows/tile);

	dim3 block(tile,tile,1);
	dim3 grid(dimx,dimy,1);
	task1c<float><<<grid,block>>>(in_,out_,rows,cols);
	cudaThreadSynchronize();
}

template<class T, int tile>
void task1d_trans(float* in_, float* out_, int rows, int cols) {
	int dimx = ceil((float)cols/tile);
	int dimy = ceil((float)rows/tile);

	dim3 block(tile,tile,1);
	dim3 grid(dimx,dimy,1);
	task1d<float,tile><<<grid,block>>>(in_,out_,rows,cols);
	cudaThreadSynchronize();
}

int main() {
	bool val;
	size_t size = COLS * ROWS;
	size_t mem_size = sizeof(float) * size;
	float* m1 = (float*) malloc(mem_size);
	float* m2 = (float*) malloc(mem_size);
	float* m3 = (float*) malloc(mem_size);

	srand(time(0));
  init_mat(m1,size);
	//TEST TASK 1.a

	unsigned long int elapsed;
	struct timeval t_start,t_end,t_diff;
	gettimeofday(&t_start, NULL);

	task1a<float>(m1,m2,ROWS,COLS);

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	printf("Task1a transpose test: %lu microseconds.\n", elapsed);

	val = validate(m1,m2,ROWS,COLS, 0.01);
	printf("Task1a transpose test: %d", val);
	//TEST TASK 1.c

	init_mat(m1,size);

	float* d1;
	cudaMalloc((void**)&d1,mem_size);

	float* d2;
	cudaMalloc((void**)&d2,mem_size);

	cudaMemcpy(d1,m1,mem_size,cudaMemcpyHostToDevice);
	gettimeofday(&t_start,NULL);

	task1c_trans<float,TILE>(d1,d2,ROWS,COLS);

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	cudaMemcpy(m3,d2,mem_size,cudaMemcpyDeviceToHost);

	printf("Task1c transpose test: %lu microseconds.\n", elapsed);

	val = validate(m1,m3,ROWS,COLS, 0.01);
	printf("Task1c transpose test: %d", val);

	//TEST TASK 1.d

	init_mat(m1,size);

	cudaMalloc((void**)&d1,mem_size);

	cudaMalloc((void**)&d2,mem_size);

	cudaMemcpy(d1,m1,mem_size,cudaMemcpyHostToDevice);
	gettimeofday(&t_start,NULL);

	task1d_trans<float,TILE>(d1,d2,ROWS,COLS);

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	cudaMemcpy(m3,d2,mem_size,cudaMemcpyDeviceToHost);

	printf("Task1d transpose test: %lu microseconds.\n", elapsed);

	val = validate(m1,m3,ROWS,COLS, 0.01);
	printf("Task1d transpose test: %d", val);

	printf("Task1a transpose test: %d", val);//TEST Task 2.c

	int num_threads = (ROWS/64)*COLS;
	int block = 256;
	int grid = num_threads / block;

	init_mat(m1,size);
	cudaMemcpy(d1,m1,mem_size,cudaMemcpyHostToDevice);
	gettimeofday(&t_start,NULL);
	task2d<<<grid,block>>>(d1,d2,num_threads);
	cudaThreadSynchronize();
	gettimeofday(&t_end,NULL);
	timeval_subtract(&t_diff,&t_end,&t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	cudaMemcpy(m3,d2,mem_size,cudaMemcpyDeviceToHost);
	printf("Task 2c transpose test: %lu microseconds.\n", elapsed);
	cudaFree(m1);
	cudaFree(m2);
	cudaFree(m3);
	cudaFree(d1);
	cudaFree(d2);
	return 0;
}
