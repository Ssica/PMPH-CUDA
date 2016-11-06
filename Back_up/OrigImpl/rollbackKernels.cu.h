#ifndef ROLLBACK_KERS
#define ROLLBACK_KERS


__global__ void explicitX(REAL dtInv, REAL* d_myResult, REAL* d_myVarX, REAL* d_myDxx, REAL* d_u, 
                          unsigned int numX, unsigned int numY, unsigned int expand) {
    
    unsigned int h = blockIdx.z;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;

    if(i < numX && j < numY) {
        d_u[h * expand + j * numX + i] = dtInv * d_myResult[h * expand + i * numY + j];

        if(i > 0) {
            d_u[h * expand + j * numX + i] += 0.5*( 0.5*d_myVarX[i * numY + j]
                                       *d_myDxx[i * 4])
                                       *d_myResult[h * expand + (i-1) * numY + j];
        }
        
        d_u[h * expand + j * numX + i] += 0.5*( 0.5*d_myVarX[i * numY + j]
                                   *d_myDxx[i * 4 + 1])
                                   *d_myResult[h * expand + i * numY + j];


        if(i < numX-1) {
            d_u[h * expand + j * numX + i] += 0.5*( 0.5*d_myVarX[i * numY + j]
                                       *d_myDxx[i * 4 + 2])
                                       *d_myResult[h * expand + (i+1) * numY + j];
        }
    }
}


__global__ void explicitY(REAL* d_myResult, REAL* d_myVarY, REAL* d_myDyy, REAL* d_v, REAL* d_u,
                          unsigned int numX, unsigned int numY, unsigned int expand) {
       
    unsigned int h = blockIdx.z;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;

    if(i < numX && j < numY) {
        d_v[h * expand +  i* numY + j] = 0.0;

        if(j > 0) {
            d_v[h * expand + i* numY + j] +=  ( 0.5*d_myVarY[i * numY+ j]
                                       *d_myDyy[j * 4 + 0] )
                                       *d_myResult[h * expand + i * numY + (j-1)];
        }
        d_v[h * expand + i* numY + j]  +=   ( 0.5*d_myVarY[i * numY +j]
                                     *d_myDyy[j * 4 + 1] )
                                     *d_myResult[h * expand + i * numY + j];
        if(j < numY-1) {
            d_v[h * expand + i* numY + j] +=  ( 0.5*d_myVarY[i * numY +j]
                                       *d_myDyy[j * 4 + 2] )
                                       *d_myResult[h * expand + i * numY + (j+1)];
        }
        d_u[h * expand + j * numX + i] += d_v[h * expand + i * numY + j]; 
    }
       
} 


__global__ void implicitX(const REAL dtInv, REAL* d_myVarX, REAL* d_myDxx, 
                                REAL* alist, REAL* blist, REAL* clist,
             const unsigned int numX, const unsigned int numY, const unsigned int expand) {

    unsigned int h = blockIdx.z;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
    
    if(i < numX && j <numY) { 
        alist[h * expand + j * numX + i] = - 0.5*(0.5* d_myVarX[ i * numY + j]*d_myDxx[i * 4 + 0]);
        blist[h * expand + j * numX + i] = dtInv - 0.5*(0.5*d_myVarX[i * numY + j]*d_myDxx[i * 4 + 1]);
        clist[h * expand + j * numX + i] = - 0.5*(0.5*d_myVarX[i * numY + j]*d_myDxx[i * 4 + 2]);
    }
}

__global__ void implicitY(const REAL dtInv, REAL* d_myVarY, REAL* d_myDyy,
                                REAL* alist, REAL* blist, REAL* clist,
             const unsigned int numX, const unsigned int numY, const unsigned int expand) {
    
    unsigned int h = blockIdx.z;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
    
    if(i < numX && j <numY) { 
        alist[h * expand + i * numY + j] = - 0.5*(0.5*d_myVarY[i * numY + j]*d_myDyy[j * 4 + 0]);
        blist[h * expand + i * numY + j] = dtInv - 0.5*(0.5*d_myVarY[i * numY + j]*d_myDyy[j * 4 + 1]);
        clist[h * expand + i * numY + j] = - 0.5*(0.5*d_myVarY[i * numY + j]*d_myDyy[j * 4 + 2]);
    }
}

#endif
