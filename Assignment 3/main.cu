#include "help.h"
#include "task1a.h"

#define ROWS 8192
#define COLS 8192
#define TILE 32

int main() {
	size_t size = COLS * ROWS;
	size_t mem_size = sizeof(float) * size;
	float* h_A = (float*) malloc(mem_size);
	float* h_B = (float*) malloc(mem_size);
	float* h_C = (float*) malloc(mem_size);

	srand(time(0));

//initializes matrix with values
	void init_matrix(float* data,int size) {
		for (int i = 0; i < size; ++i) {
			//inserts random values from 1-100
			h_A[i] = (float)(rand()%100+1);
		}
	}

	//SEQUENTIAL TRANSPOSE

	unsigned long int elapsed;
	struct timeval t_start,t_end,t_diff;
	gettimeofday(&t_start, NULL);

	task1a<float>(h_A,h_B,ROWS,COLS);

	gettimeofday(&t_end, NULL);

	timeval_subtract(&t_diff, &t_end, &t_start);

	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	printf("Sequential transpose test: %lu microseconds.\n", elapsed);

	printf("Sequential transpose test:\n");
	valid_trans<float>(h_A,h_B,ROWS,COLS);
	return 0;
}
