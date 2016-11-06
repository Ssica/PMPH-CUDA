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
/*
__global__ void firstTridagLoop(REAL* d_alist, REAL* d_blist, REAL* d_clist, REAL* d_u, 
			   const unsigned numX, REAL* d_u2, REAL* d_yy, 
			   const unsigned int numY, const unsigned numZ, 
   			   const unsigned int expand, const unsigned int expand_Z){
    unsigned int h = blockIdx.z;
    unsigned int j = blockIdx.x*blockDim.x+threadIdx.x;

    if(j < numY){
	    int i;
	    REAL beta;
	    
            d_u2[h * expand + j * numX + 0] = d_u[h * expand + j * numX + 0];
            d_yy[h * numZ + 0] = d_blist[h * expand_Z + j * numX  + 0];
	    
	    for(i=1; i < numX; i++){
		beta = d_alist[h * expand_Z + j * numX + i]/d_yy[h * numZ + (i-1)];
		
		d_yy[h * numZ + i] = d_blist[h * expand_Z + j * numX + i] - 
				     beta*d_clist[h * expand_Z + j * numX + (i-1)];

		d_u2[h * expand + j * numX + i] = d_u[h * expand + j * numX + i] - 
				     beta*d_u2[h * expand + j * numX + (i-1)];
	
	 
	    }

	    d_u2[h * expand + j * numX + (numX-1)] = d_u2[h * expand + j * numX + (numX-1)]
					/ d_yy[h * numZ + (numX-1)];

	    for(i = numX-2; i>=0; i--){
		d_u2[h * expand + j * numX + i] = (d_u2[h * expand + j * numX + i] - 
					d_clist[h * expand_Z + j * numX + i] * 
					d_u2[h * expand + j * numX + (i+1)]) /
					d_yy[h * numZ + i];

	    }

	}
}

*/

__global__ void firstTridagLoop(REAL* d_alist, REAL* d_blist, REAL* d_clist, REAL* d_u,  REAL* d_yy, 
			   const unsigned int numZ, const unsigned numZ_){
    unsigned int h = blockIdx.z;
    unsigned int j = blockIdx.x*blockDim.x+threadIdx.x;
    printf("h: %d, r: %f\n", h, d_u[h * numZ * numZ_ + j * numZ]);

    if(j < numZ){
        unsigned int expand = numZ * numZ_ * h;

        REAL* a = d_alist + (expand + j*numZ_);
        REAL* b = d_blist + (expand + j*numZ_);
        REAL* c = d_clist + (expand + j*numZ_);
        REAL* r = d_u + (expand + j*numZ_);
        const int n = numZ_;
        REAL* u = d_u + (expand + j*numZ_);
        REAL* uu = d_yy + (h*numZ);


        int i;
        REAL beta;

        u[0] = r[0];
        uu[0] = b[0];

        for(i=1; i<n; i++) {
            beta = a[i] / uu[i-1];
            uu[i] = b[i] - beta*c[i-1];
            u[i]  = r[i] - beta*u[i-1];
        }

        u[n-1] = u[n-1] / uu[n-1];
        for(i=n-2; i>=0; i--) {
            u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
        }

    }
}

__global__ void ylistKernel(REAL dtInv, REAL* d_ylist, REAL* d_u, REAL* d_v, 
                           const unsigned int numZ, const unsigned int numZ_) {
    unsigned int h = blockIdx.z;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;

    if(i < numZ_ && j < numZ) {
        unsigned int expand = h * numZ * numZ_;
        d_ylist[h * expand + i * numZ + j] = dtInv*d_u[h* expand + j * numZ_ + i]
                                   - 0.5*d_v[h * expand + i * numZ + j];
    }
}






#endif
