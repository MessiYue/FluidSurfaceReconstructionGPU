#include "Kernel.cuh"

void initSimParam(SimParam *params)
{
	checkCudaErrors(cudaMemcpyToSymbol(simParam, params, sizeof(SimParam)));
	cudaDeviceSynchronize();
}

uint ThrustExclusiveScanWrapper(uint* output, uint* input, uint numElements)
{
	//! exclusive prefix sum.
	thrust::exclusive_scan(
		thrust::device_ptr<uint>(input),
		thrust::device_ptr<uint>(input + numElements),
		thrust::device_ptr<uint>(output));
	cudaDeviceSynchronize();

	uint lastElement = 0;
	uint lastElementScan = 0;
	checkCudaErrors(cudaMemcpy((void *)&lastElement, (void *)(input + numElements - 1), 
		sizeof(uint), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((void *)&lastElementScan, (void *)(output + numElements - 1),
		sizeof(uint), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	uint sum = lastElement + lastElementScan;
	return sum;
}