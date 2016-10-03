template<class T>
__global__ void task1c(float* in_, float* out_, int rows, int cols) {

	int i = blockIdx.y*blockDim.y+threadIdx.y;
	int j = blockIdx.x*blockDim.x+threadIdx.x;

	if ((i >= rows) || (j >= cols)) {
		return;
	} else {
		out_[j*rows+i] = in_[i*cols+j];
	}
}
