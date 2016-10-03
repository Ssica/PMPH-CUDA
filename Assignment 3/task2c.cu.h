__global__ void task2c(float* A, float* B, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    bool t= true;
    while(t)
    {
        gid = gid * 64;
        float tmpB = A[gid] * A[gid];
        B[gid] = tmpB;
        for (int j = 1; j < 64; j++) {
            float tmpA = A[gid + j];
            float accum = sqrt(tmpB) + tmpA * tmpA;
            B[gid + j] = accum;
            tmpB = accum;
            if(gid < N){
                t = false;
            }
        }
    }
    return;
}