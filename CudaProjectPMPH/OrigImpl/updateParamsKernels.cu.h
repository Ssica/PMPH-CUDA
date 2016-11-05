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


__global__ void setParamsKer(const unsigned int numX, const unsigned int numY, REAL* myX, REAL* myResult)
{
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int h = blockIdx.z;
    
    myResult[h * numX * numY + i *  numY + j] = max(myX[i]-0.001*h, (REAL)0.0);

}

__global__ void implicitX(const unsigned int numX, const unsigned int numX, const REAL dtInv,
                          REAL* d_myVarX, REAL* d_myDxx, REAL* alist, REAL* blist, REAL* clist){

    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
    
    alist[ j * numX + i] = - 0.5*(0.5* d_myVarX[ i * numY + j]*d_myDxx[i * 4 + 0]);
    blist[ j * numX + i] = dtInv - 0.5*(0.5*d_myVarX[i * numY + j]*d_myDxx[i * 4 + 1]);
    clist[ j * numX + i] = - 0.5*(0.5*d_myVarX[i * numY + j]*d_myDxx[i * 4 + 2]);
}
#endif
