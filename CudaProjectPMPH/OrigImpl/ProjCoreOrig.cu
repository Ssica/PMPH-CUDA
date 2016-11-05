#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"
#include "updateParamsKernels.cu.h"

#define BLOCK_DIM 8
void GPUimplicitX(PrivGlobs& globs, REAL* alist, REAL* blist, REAL* clist, const unsigned g){

  unsigned int block_dim = BLOCK_DIM;
  unsigned int numZ = max(globs.numX, globs.numY);
  REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

  dim3 threadsPerBlock(block_dim, block_dim, 1);
  dim3 num_blocks(ceil((float)globs.numX/block_dim), ceil((float)globs.numY/block_dim));
  REAL* d_myVarX, *d_myDxx, *d_alist, *d_blist, *d_clist;

  cudaMalloc((void**)&d_myVarX, globs.numX*globs.numY*sizeof(REAL));
  cudaMalloc((void**)&d_myDxx, globs.numX*4*sizeof(REAL));
  cudaMalloc((void**)&d_alist, numZ*numZ*sizeof(REAL));
  cudaMalloc((void**)&d_blist, numZ*numZ*sizeof(REAL));
  cudaMalloc((void**)&d_clist, numZ*numZ*sizeof(REAL));

  cudaMemcpy(d_myVarX, globs.myVarX, globs.numX*globs.numY*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_myDxx, globs.myDxx, globs.numX*4*sizeof(REAL), cudaMemcpyHostToDevice);

  implicitX<<<num_blocks, threadsPerBlock>>>(globs.numX, globs.numY, dtInv, d_myVarX, d_myDxx,
                                             d_alist, d_blist, d_clist);
  cudaMemcpy(alist, d_alist, numZ*numZ*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(blist, d_blist, numZ*numZ*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(clist, d_clist, numZ*numZ*sizeof(REAL), cudaMemcpyDeviceToHost);

  cudaFree(d_myVarX);
  cudaFree(d_myDxx);
  cudaFree(d_alist);
  cudaFree(d_blist);
  cudaFree(d_clist);

}

void GPUimplicitY(PrivGlobs& globs, REAL* alist, REAL* blist, REAL* clist, const unsigned g){

  unsigned int block_dim = BLOCK_DIM;
  unsigned int numZ = max(globs.numX, globs.numY);
  REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

  dim3 threadsPerBlock(block_dim, block_dim, 1);
  dim3 num_blocks(ceil((float)globs.numX/block_dim), ceil((float)globs.numY/block_dim));
  REAL* d_myVarY, *d_myDyy, *d_alist, *d_blist, *d_clist;

  cudaMalloc((void**)&d_myVarY, globs.numX*globs.numY*sizeof(REAL));
  cudaMalloc((void**)&d_myDyy, globs.numY*4*sizeof(REAL));
  cudaMalloc((void**)&d_alist, numZ*numZ*sizeof(REAL));
  cudaMalloc((void**)&d_blist, numZ*numZ*sizeof(REAL));
  cudaMalloc((void**)&d_clist, numZ*numZ*sizeof(REAL));

  cudaMemcpy(d_myVarY, globs.myVarY, globs.numX*globs.numY*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_myDyy, globs.myDyy, globs.numY*4*sizeof(REAL), cudaMemcpyHostToDevice);

  implicitY<<<num_blocks, threadsPerBlock>>>(globs.numX, globs.numY, dtInv, d_myVarY, d_myDyy,
                                             d_alist, d_blist, d_clist);
  cudaMemcpy(alist, d_alist, numZ*numZ*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(blist, d_blist, numZ*numZ*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(clist, d_clist, numZ*numZ*sizeof(REAL), cudaMemcpyDeviceToHost);

  cudaFree(d_myVarY);
  cudaFree(d_myDyy);
  cudaFree(d_alist);
  cudaFree(d_blist);
  cudaFree(d_clist);

}

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
        for(unsigned i=0;i<globs.numX;++i)
            for(unsigned j=0;j<globs.numY;++j) {
                globs.myVarX[i * globs.numY + j] =
                                              exp(2.0*(  beta*log(globs.myX[i])
                                            + globs.myY[j]
                                            - 0.5*nu*nu*globs.myTimeline[g] )
                                        );
                globs.myVarY[i * globs.numY + j] =
                                              exp(2.0*(  alpha*log(globs.myX[i])
                                            + globs.myY[j]
                                            - 0.5*nu*nu*globs.myTimeline[g] )
                                        ); // nu*nu
           }

}

void setPayoff(PrivGlobs& globs )
{
    //REAL* payoff = (REAL*) malloc(globs.outer * globs.numX*sizeof(REAL));
    for(unsigned h=0;h<globs.outer;h++)
	    for(unsigned i=0;i<globs.numX;++i) {
            for(unsigned j=0;j<globs.numY;++j)
		        globs.myResult[h * globs.numX * globs.numY + i * globs.numY + j] =
		                                          max(globs.myX[i]-0.001*h, (REAL)0.0);
            }


}

inline void tridag(
    REAL*   a,   // size [n]
    REAL*   b,   // size [n]
    REAL*   c,   // size [n]
    REAL*   r,   // size [n]
    const int             n,
          REAL*   u,   // size [n]
          REAL*   uu   // size [n] temporary
) {
    int    i;//, offset;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

#if 1
    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
    }
#else
    // Hint: X) can be written smth like (once you make a non-constant)
    for(i=0; i<n; i++) a[i] =  u[n-1-i];
    a[0] = a[0] / uu[n-1];
    for(i=1; i<n; i++) a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
    for(i=0; i<n; i++) u[i] = a[n-1-i];
#endif
 }


void
rollback( const unsigned g, const unsigned h, PrivGlobs& globs) {
    unsigned numX = globs.numX,
             numY = globs.numY;

    unsigned numZ = max(numX,numY);
    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
/*
    vector< vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector< vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    vector< vector<REAL> > alist(numZ, vector<REAL>(numZ));
    vector< vector<REAL> > blist(numZ, vector<REAL>(numZ));
    vector< vector<REAL> > clist(numZ, vector<REAL>(numZ));
    vector< vector<REAL> > ylist(numZ, vector<REAL>(numZ));
    vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]
*/
    REAL* u = (REAL*) malloc(numY*numX*sizeof(REAL));
    REAL* v = (REAL*) malloc(numY*numX*sizeof(REAL));
    REAL* alist = (REAL*) malloc(numZ*numZ*sizeof(REAL));
    REAL* blist = (REAL*) malloc(numZ*numZ*sizeof(REAL));
    REAL* clist = (REAL*) malloc(numZ*numZ*sizeof(REAL));
    REAL* ylist = (REAL*) malloc(numZ*numZ*sizeof(REAL));
    REAL* yy = (REAL*) malloc(numZ*sizeof(REAL));

    unsigned int expand = numX * numY;

    //	explicit x

        for(i=0;i<numX;i++) {
            for(j=0;j<numY;j++) {
                u[j * numX + i] = dtInv*globs.myResult[h * expand + i * numY + j];

                if(i > 0) {
                  u[j * numX + i] += 0.5*( 0.5*globs.myVarX[i * numY + j]
                                           *globs.myDxx[i * 4 + 0])
                                           * globs.myResult[h * expand + (i-1) * numY + j];
                }
                u[j * numX + i]  +=  0.5*( 0.5*globs.myVarX[i * numY + j]
                                           *globs.myDxx[i * 4 + 1])
                                           *globs.myResult[h * expand + i * numY + j];
                if(i < numX-1) {
                  u[j * numX + i] += 0.5*( 0.5*globs.myVarX[i * numY + j]
                                           *globs.myDxx[i * 4 + 2])
                                           * globs.myResult[h * expand + (i+1) * numY + j];
                }
            }
        }


    //	explicit y
        for(j=0;j<numY;j++)
        {
            for(i=0;i<numX;i++) {
                v[ i* numY + j] = 0.0;

                if(j > 0) {
                  v[ i* numY + j] +=  ( 0.5*globs.myVarY[i * numY+ j]
                                               *globs.myDyy[j * 4 + 0] )
                                               *globs.myResult[h * expand + i * numY + (j-1)];
                }
                v[ i* numY + j]  +=   ( 0.5*globs.myVarY[i * numY +j]
                                               *globs.myDyy[j * 4 + 1] )
                                               *globs.myResult[h * expand + i * numY + j];
                if(j < numY-1) {
                  v[ i* numY + j] +=  ( 0.5*globs.myVarY[i * numY +j]
                                               *globs.myDyy[j * 4 + 2] )
                                               *globs.myResult[h * expand + i * numY + (j+1)];
                }
                u[ + j * numX + i] += v[ i * numY + j];
            }
        }

/*
    //	implicit x
        for(j=0;j<numY;j++) {
            for(i=0;i<numX;i++) {  // here a,b,c should have size [numX]
                alist[ j * numX + i] = - 0.5*(0.5*
                     globs.myVarX[ i * numY + j]*globs.myDxx[i * 4 + 0]);
                blist[ j * numX + i] = dtInv - 0.5*(0.5*
                     globs.myVarX[i * numY + j]*globs.myDxx[i * 4 + 1]);
                clist[ j * numX + i] = - 0.5*(0.5*
                     globs.myVarX[i * numY + j]*globs.myDxx[i * 4 + 2]);
            }
        //tridagPar(alist[j],blist[j],clist[j],u[j],numX,u[j],yy);
        // here yy should have size [numX]
        }
*/
        GPUimplicitX(globs, alist, blist, clist, g);
        for(j=0;j<numY;j++) {
            tridag(&alist[ j*numX],&blist[ j*numX],
                   &clist[ j*numX],&u[ j*numX],
                    numX,&u[j*numX],yy);
        }

    //	implicit y
        for(i=0;i<numX;i++) {
            for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                alist[ i * numY + j] = - 0.5*(0.5*globs.myVarY[i * numY + j]
                                                  *globs.myDyy[j * 4 + 0]);
                blist[ i * numY + j] = dtInv - 0.5*(0.5
                                                  *globs.myVarY[i * numY + j]
                                                  *globs.myDyy[j * 4 + 1]);
                clist[ i * numY + j] = - 0.5*(0.5
                                                  *globs.myVarY[i * numY + j]
                                                  *globs.myDyy[j * 4 + 2]);
            }
        }


        // here yy should have size [numY]
      //  GPUimplicitY(globs, alist, blist, clist, g);
        for(i=0;i<numX;i++) {
            for(j=0;j<numY;j++) {
                ylist[ i * numY + j] = dtInv*u[ + j * numX + i]
                                             - 0.5*v[ i * numY + j];
            }
        }

        for(i=0;i<numX;i++) {
            tridag(&alist[ i*numY],&blist[ i*numY],
                   &clist[ i*numY],&ylist[ i*numY],
                    numY,&globs.myResult[h * expand + i*numY],yy);
        }

}

REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike,
                const REAL t,
                const REAL alpha,
                const REAL nu,
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
) {
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx, numX);
    initOperator(globs.myY,globs.myDyy, numY);

    setPayoff(globs);
    for(int i = numT-2;i>=0;--i)
    {
        updateParams(i,alpha,beta,nu,globs);
        //rollback(i, 2, globs);
    }

    return globs.myResult[globs.myXindex * numY + globs.myYindex];
}

void GPUupdateParams(const unsigned g, const REAL alpha, const REAL beta,
                                         const REAL nu, PrivGlobs& globs)
{

    unsigned int block_dim = BLOCK_DIM;
    dim3 threadsPerBlock(block_dim, block_dim, 1);
    dim3 num_blocks(ceil((float)globs.numX/block_dim), ceil((float)globs.numY/block_dim),1);

    REAL *d_myVarX, *d_myVarY, *d_myX, *d_myY, *d_myTimeline;
    cudaMalloc((void**)&d_myVarX, globs.numX*globs.numY*sizeof(REAL));
    cudaMalloc((void**)&d_myVarY, globs.numX*globs.numY*sizeof(REAL));
    cudaMalloc((void**)&d_myX, globs.numX*sizeof(REAL));
    cudaMalloc((void**)&d_myY, globs.numY*sizeof(REAL));
    cudaMalloc((void**)&d_myTimeline, globs.numT*sizeof(REAL));

    cudaMemcpy(d_myVarX, globs.myVarX, globs.numX*globs.numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myVarY, globs.myVarY, globs.numX*globs.numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myX, globs.myX, globs.numX*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myY, globs.myY, globs.numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myTimeline, globs.myTimeline,
               globs.numT*sizeof(REAL), cudaMemcpyHostToDevice);

    updateParamsKer<<<num_blocks, threadsPerBlock>>>(g, alpha, beta, nu, globs.numX, globs.numY,
                                   d_myX, d_myY, d_myVarX, d_myVarY, d_myTimeline);

    cudaThreadSynchronize();

    cudaMemcpy(globs.myVarX, d_myVarX, globs.numX*globs.numY*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(globs.myVarY, d_myVarY, globs.numX*globs.numY*sizeof(REAL), cudaMemcpyDeviceToHost);


    cudaFree(d_myVarX);
    cudaFree(d_myVarY);
    cudaFree(d_myX);
    cudaFree(d_myY);
    cudaFree(d_myTimeline);


}

void GPUsetParams(PrivGlobs& globs)
{

    unsigned int block_dim = BLOCK_DIM;
    dim3 threadsPerBlock(block_dim, block_dim, 1);
    dim3 num_blocks(ceil((float)globs.numX/block_dim), ceil((float)globs.numY/block_dim),globs.outer);

    REAL *d_myX, *d_myResult;
    cudaMalloc((void**)&d_myResult, globs.outer*globs.numX*globs.numY*sizeof(REAL));
    cudaMalloc((void**)&d_myX, globs.numX*sizeof(REAL));

    cudaMemcpy(d_myResult, globs.myResult, globs.outer*globs.numX*globs.numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myX, globs.myX, globs.numX*sizeof(REAL), cudaMemcpyHostToDevice);

    setParamsKer<<<num_blocks, threadsPerBlock>>>(globs.numX, globs.numY, d_myX, d_myResult);

    cudaThreadSynchronize();

    cudaMemcpy(globs.myResult, d_myResult, globs.outer*globs.numX*globs.numY*sizeof(REAL), cudaMemcpyDeviceToHost);

    cudaFree(d_myX);
    cudaFree(d_myResult);


}

void   run_OrigCPU(
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t,
                const REAL&           alpha,
                const REAL&           nu,
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
) {

    PrivGlobs globs(numX, numY, numT, outer);
/*
    for(unsigned i = 0; i < outer;++i) {
        strike[i] = 0.001*i;
        //PrivGlobs globs(numX, numY, numT, outer);
        //globslist[i] = globs;

    }*/

    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx, numX);
    initOperator(globs.myY,globs.myDyy, numY);

    //setPayoff(globs);
    GPUsetParams(globs);
    for(int h=0; h<outer; h++) {
        for(int i = numT-2;i>=0;--i)
        {
            GPUupdateParams(i,alpha,beta,nu,globs);
            rollback(i, h, globs);
        }
    }

    for(unsigned i = 0; i < outer; ++i){
        res[i] = globs.myResult[i * numX * numY + globs.myXindex * numY + globs.myYindex];
    }
/*
    for( unsigned i = 0; i < outer; ++ i ) {
        res[i] = value( globslist[i], s0, strike[i], t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }*/
}

//#endif // PROJ_CORE_ORIG
