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


__global__ void setParamsKer(REAL d_numX ,REAL d_numY, REAL* d_myX, REAL* d_myResult)
{
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int outerIdx = blockIdx.z;
    
    d_myResult[outerIdx * d_numX * d_numY + i * d_numY + j] =
                                    max(d_myX[i]-0.001*outerIdx, (REAL)0.0);

}

#endif
