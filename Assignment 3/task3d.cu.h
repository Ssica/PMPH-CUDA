template<class T>
__global__ void task3d(float* m1, float*m2, float* m_out, int col1, int row1, int col2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((i < col2) && (j < row1)) {
        T res = 0;
        for(int k = 0; k < col1; k++) {
            res = res + A[j*col1+k] * B[k*col2+i];
            }
        res[j*cols2+i] = res;
        }
    else {
        return;
    }
}