template<class T>
__global__ void task3c(float* m1, float*m2, float* m_out, int row1, int col1, int col2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((i < col2) && (j < row1)) {
        T res = 0;
        for(int k = 0; k < col1; k++) {
            res = res + m1[j*col1+k] * m2[k*col2+i];
            }
        res[j*col2+i] = res;
        }
    else {
        return;
    }
}