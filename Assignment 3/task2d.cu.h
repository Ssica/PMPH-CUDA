__global__ void task2d(float* A, float* B, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    bool t= true;
    if (gid < N) {
        float tmpB = A[gid] * A[gid];
        B[gid] = tmpB;
        gid = gid + N;
        for (int j = 1; j < 64; j++) {
            gid = gid + N;
            float tmpA = A[gid + j];
            float accum = sqrt(tmpB) + tmpA * tmpA;
            B[gid + j] = accum;
            tmpB = accum;
        }
    }
    return;
}