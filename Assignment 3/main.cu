#include "help.h"
#include "task1a.h"
#include "task1c.cu.h"
#include "task1d.cu.h"
#include "task2c.cu.h"
#include "task2d.cu.h"
#include "task3a.h"
#include "task3c.cu.h"

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
	printf("Task1a transpose test: %d \n", val);
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
	printf("Task1c transpose test: %d \n", val);

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
	printf("Task1d transpose test: %d \n", val);

	//TEST Task 2.c

	int num_threads = (ROWS/64)*COLS;
	int block = 256;
	int grid = num_threads / block;

	init_mat(m1,size);
	cudaMemcpy(d1,m1,mem_size,cudaMemcpyHostToDevice);
	gettimeofday(&t_start,NULL);
	task2c<<<grid,block>>>(d1,d2,num_threads);
	cudaThreadSynchronize();
	gettimeofday(&t_end,NULL);
	timeval_subtract(&t_diff,&t_end,&t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	cudaMemcpy(m3,d2,mem_size,cudaMemcpyDeviceToHost);
	printf("Task 2c transpose test: %lu microseconds.\n", elapsed);

	//TEST Task 2.d

	init_mat(m1,size);
	cudaMemcpy(d1,m1,mem_size,cudaMemcpyHostToDevice);
	gettimeofday(&t_start,NULL);
	task2d<<<grid,block>>>(d1,d2,num_threads);
	cudaThreadSynchronize();
	gettimeofday(&t_end,NULL);
	timeval_subtract(&t_diff,&t_end,&t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	cudaMemcpy(m3,d2,mem_size,cudaMemcpyDeviceToHost);
	printf("Task 2d transpose test: %lu microseconds.\n", elapsed);

	//TEST Task 3.a and 3.c

	int row1 = 1024;
	int col1 = 1024;
	int row2 = 1024;
	int col2 = 1024;

	int res_mem = row1*col2*sizeof(float);
	int mem_size1 = row1*col1*sizeof(float);
	int mem_size2 = row2*col2*sizeof(float);

  float* m1_ = (float*) malloc(mem_size1);
	float* m2_ = (float*) malloc(mem_size2);
	float* m3_ = (float*) malloc(mem_size1);
	float* res_ = (float*) malloc(res_mem);
	init_mat(m1_, row1);
	init_mat(m2_, row2);

	float* d_m1;
	float* d_m2;
	float* d_res;

	cudaMalloc((void**)&d_m1,mem_size1);
	cudaMalloc((void**)&d_m2,mem_size2);
	cudaMalloc((void**)&d_res,res_mem);

	cudaMemcpy(d_m1,m1_,mem_size1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2,m2_,mem_size2,cudaMemcpyHostToDevice);

	gettimeofday(&t_start,NULL);

	m3_ = task3a(m1_,m2_,row1,col1,col2);

	gettimeofday(&t_end,NULL);
	timeval_subtract(&t_diff,&t_end,&t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
	printf("Task 3a: %lu microseconds.\n", elapsed);
	gettimeofday(&t_start,NULL);

	task3c<float><<<grid,block>>>(d_m1,d_m2,d_res,row1,col1,col2);

	gettimeofday(&t_end,NULL);
	timeval_subtract(&t_diff,&t_end,&t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

  cudaMemcpy(res_,d_res,res_mem,cudaMemcpyDeviceToHost);
	printf("Task 3d: %lu microseconds.\n", elapsed);
  val = validate(m3_,res_,row1,col2, 0.01);
	printf("Task3c matrix mult test: %d", val);
	cudaFree(m1);
	cudaFree(m2);
	cudaFree(m3);
	cudaFree(d1);
	cudaFree(d2);
	return 0;
}
