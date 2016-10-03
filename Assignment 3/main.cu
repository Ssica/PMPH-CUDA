#include "help.h"
#include "task1a.h"
#include "task1c.cu.h"
#include "task1d.cu.h"

#define ROWS 1024
#define COLS 1024
#define TILE 32

template<class T, int tile>
void task1c_trans(float* in_, float* out_, int rows, int cols) {
	int dimy = ceil((float)rows/tile);
	int dimx = ceil((float)cols/tile);

	dim3 block(tile,tile,1);
	dim3 grid(dimx,dimy,1);
	par_transpose<float><<<grid,block>>>(in_,out_,rows,cols);
	cudaThreadSynchronize();
}

template<class T, int tile>
void task1d_trans(float* in_, float* out_, int rows, int cols) {
	int dimx = ceil((float)cols/tile);
	int dimy = ceil((float)rows/tile);

	dim3 block(tile,tile,1);
	dim3 grid(dimx,dimy,1);
	par_tiled_transpose<float,tile><<<grid,block>>>(in_,out_,rows,cols);
	cudaThreadSynchronize();
}
int main() {
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

	printf("Task1a transpose test:\n");
	validate(m1,m2,ROWS,COLS, 0.01);

	//TEST TASK 1.C

	init_mat(m1,size);

	float* d1;
	cudaMalloc((void**)&d1,mem_size);

	float* d2;
	cudaMalloc((void**)&d2,mem_size);

	cudaMemcpy(d1,m1,mem_size,cudaMemcpyHostToDevice);
	gettimeofday(&t_start,NULL);

	task1c_trans<float,TILE>(d1,d2,ROWS,COLS,1);

	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	cudaMemcpy(m3,d2,mem_size,cudaMemcpyDeviceToHost);

	printf("Task1c transpose test: %lu microseconds.\n", elapsed);

	printf("Task1c transpose test:\n");
	validate(m1,m2,ROWS,COLS, 0.01);

	return 0;
}
