#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ScanHost.cu.h"


int scanIncTest(bool is_segmented) {
    const unsigned int num_threads = 8353455;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1;
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    { // calling inclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        if(is_segmented)
            sgmScanInc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
        else
            scanInc< Add<int>,int > ( block_size, num_threads, d_in, d_out );

        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Inclusive on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    } else {
        for(int i=0; i<num_threads; i++) {
            accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    }

    if(success) printf("\nScan Inclusive +   VALID RESULT!\n");
    else        printf("\nScan Inclusive + INVALID RESULT!\n");


    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}
int scanExclTest(bool is_segmented) {
    const unsigned int num_threads = 8353455;
    const unsigned int block_size  = 512;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1;
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    { // calling inclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        if(is_segmented)
            sgmScanExcl< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
        else
            scanExcl< Add<int>,int > ( block_size, num_threads, d_in, d_out );

        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Exclusive on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            else accum += 1;

            if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }
    } else {
        for(int i=0; i<num_threads; i++) {

	    if ( accum != h_out[i] ) {
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
	    accum += 1; //we iterate after check because exclusive scan starts with ne, in this case 0


        }
    }

    if(success) printf("\nScan Exclusive +   VALID RESULT!\n\n");
    else        printf("\nScan Exclusive + INVALID RESULT!\n\n");


    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}

void msspTest(){
  int block_size = 512;
  int len = 9000;
  int mem_size = len*sizeof(MyInt4);

  int num_blocks = ( (len % block_size) == 0) ?
                    len / block_size     :
                    len / block_size + 1 ;

  int* h_in = (int*)malloc(len*sizeof(int));
  for(int i = 0; i < len; i++) {
          h_in[i] = 1;
    }
  int* d_in;
  MyInt4* d_inplift;
  MyInt4* d_out;
  MyInt4 res(0,0,0,0);
  MyInt4 *res_ = &res;

  cudaMalloc((void**)&d_in,len*sizeof(int));
  cudaMalloc((void**)&d_inplift,mem_size);
  cudaMalloc((void**)&d_out,mem_size);

  cudaMemcpy(d_in,h_in,len*sizeof(int),cudaMemcpyHostToDevice);

  unsigned long int elapsed;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);
  msspTrivialMap<<<num_blocks,block_size>>>(d_in,d_inplift,len);
  cudaThreadSynchronize();
  scanInc<MsspOp,MyInt4>(block_size,len,d_inplift, d_out);
  cudaThreadSynchronize();
  cudaMemcpy(res_,d_out+len-1,sizeof(MyInt4),cudaMemcpyDeviceToHost);

  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
  printf("mssp runs in: %d microseconds \n", elapsed);

  if(res.x == len) { printf("mssp: VALID.\n"); }
  else { printf("mssp: INVALID.\n"); }


}
int main(int argc, char** argv) {
    scanIncTest(true);
    scanIncTest(true);
    scanIncTest(false);
    scanExclTest(false);
    scanExclTest(true);
    msspTest();
}
