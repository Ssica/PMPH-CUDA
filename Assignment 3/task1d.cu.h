template<class T, int TILE>
__global__ void task1d(float* in_, float* out_, int rows, int cols) {
    
    __shared__ float tile[TILE][TILE+1];
    
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    
    int i = blockIdx.y*TILE + tidy;
    int j = blockIdx.x*TILE + tidx;
    
    if ((i < rows) && (j < cols)) {
        tile[tidy][tidx] = in_[i*cols+j];
    }
    
    __syncthreads();
    
    i = blockIdx.y*TILE + threadIdx.x;
    j = blockIdx.x*TILE + threadIdx.y;
    
    if ((i < rows) && (j < cols)) {
        out_[j*rows+i] = tile[tidx][tidy];
    }
}