#include "help.h"
#include "task1a.h"

#define ROWS 8192
#define COLS 8192
#define TILE 32

int main() {
	size_t size = COLS * ROWS;
	size_t mem_size = sizeof(float) * size;
	float* m1 = (float*) malloc(mem_size);
	float* m2 = (float*) malloc(mem_size);
	float* m3 = (float*) malloc(mem_size);

	srand(time(0));
  init_mat(m1,size);

	//SEQUENTIAL TRANSPOSE

	unsigned long int elapsed;
	struct timeval t_start,t_end,t_diff;
	gettimeofday(&t_start, NULL);

	task1a<float>(m1,m2,ROWS,COLS);

	gettimeofday(&t_end, NULL);

	timeval_subtract(&t_diff, &t_end, &t_start);

	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

	printf("Sequential transpose test: %lu microseconds.\n", elapsed);

	printf("Sequential transpose test:\n");
	validate(m1,m2,ROWS,COLS);
	return 0;
}
