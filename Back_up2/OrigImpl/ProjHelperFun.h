#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

using namespace std;


struct PrivGlobs {

    //	grid
    REAL*        myX;        // [numX]
    REAL*        myY;        // [numY]
    REAL*        myTimeline; // [numT]
    unsigned            myXindex;  
    unsigned            myYindex;
    
    unsigned int numX;
    unsigned int numY;
    unsigned int numT;

    unsigned int outer;
    unsigned int h;

    //	variable
    REAL* myResult; // [numX][numY]

    //	coeffs
    REAL*   myVarX; // [numX][numY]
    REAL*   myVarY; // [numX][numY]

    //	operators
    REAL*   myDxx;  // [numX][4]
    REAL*   myDyy;  // [numY][4]

    PrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT,
                const unsigned int& outer_
               ) {
        this->myX = (REAL*) malloc(numX*sizeof(REAL)); 
        this->myDxx = (REAL*) malloc(4*numX*sizeof(REAL)); 
        
        this->numX = numX;
        this->numY = numY;
        this->numT = numT;

        this->outer = outer_;

        this->myY = (REAL*) malloc(numY*sizeof(REAL));
        this->myDyy = (REAL*) malloc(4*numY*sizeof(REAL));
        
        this->myTimeline = (REAL*) malloc(numT*sizeof(REAL));

        this->  myVarX = (REAL*) malloc(numY*numX*sizeof(REAL)); //expanded by outer
        this->  myVarY = (REAL*) malloc(numY*numX*sizeof(REAL)); //expanded by outer
        this->myResult = (REAL*) malloc(outer*numY*numX*sizeof(REAL)); //expanded by outer
      
    }
} __attribute__ ((aligned (128)));


void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs   
            );

void initOperator(  const REAL* x, 
                    REAL* Dxx, const unsigned int x_size
                 );

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void setPayoff(PrivGlobs& globs );

void tridag(
    REAL*   a,   // size [n]
    REAL*   b,   // size [n]
    REAL*   c,   // size [n]
    REAL*   r,   // size [n]
    const int             n,
          REAL*   u,   // size [n]
          REAL*   uu   // size [n] temporary
);

void rollback( const unsigned g, PrivGlobs& globs );

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
            );

void run_OrigCPU(  
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
            );

#endif // PROJ_HELPER_FUNS
