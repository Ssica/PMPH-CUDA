#ifndef PROJ_KERS
#define PROJ_KERS

#include "Constants.h"


__global__ void updateParamsKer(const unsigned g, const REAL alpha, const REAL beta, const REAL nu,
                                unsigned int d_numX, unsigned int d_numY, 
                                REAL* d_myX, REAL* d_myY, REAL* d_myVarX, REAL* d_myVarY, 
                                REAL* d_myTimeline){

    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;    



    if(i < d_numX && j < d_numY){
    //printf("i: %d, j: %d, k: %d, bx: %d, by: %d, bz: %d\n", i, j, k, blockIdx.x, blockIdx.y, blockIdx.z);
    	d_myVarX[i * d_numY + j] = exp(2.0*(beta*log(d_myX[i]) + 
                             d_myY[j] - 
                             0.5*nu*nu*d_myTimeline[g]));
    	
        //REAL* a = exp(2.0*(beta*log(d_myX[i]) + d_myY[j] - 0.5*nu*nu*d_myTimeline[g]));
        //printf("%\n", d_numX);
    

        d_myVarY[i * d_numY + j] = exp(2.0*(alpha*log(d_myX[i]) + 
                             d_myY[j] - 
                             0.5*nu*nu*d_myTimeline[g]));
    
    }

}


__global__ void setParamsKer(const REAL numX, const REAL numY, REAL* myX, REAL* myResult,
                             const unsigned int block_dim)
{
    unsigned int i = block_dim*blockDim.x+threadIdx.x;
    unsigned int j = block_dim*blockDim.y+threadIdx.y;
    unsigned int h = blockIdx.z;
    
    myResult[h * numX * numY + i * numY + j] = (REAL) max(myX[i]-0.001*h, (REAL)0.0);

}

#endif
