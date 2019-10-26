#include "ReconstructorGPUAkinci12.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CudaUtils.h"
#include "ReconstructionCUDA.cuh"

ReconstructorGPUAkinci12::ReconstructorGPUAkinci12(
	const std::string & directory,
	const std::string & filePattern,
	unsigned int from, unsigned int to) : 
	ReconstructorGPU(directory, filePattern, from, to) {}

void ReconstructorGPUAkinci12::onBeginFrame(unsigned int frameIndex)
{
	//ReconstructorGPU::onBeginFrame(frameIndex);

	CUDA_CREATE_GRID_3D(mDeviceIsSurfaceGrid, mScalarFieldGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(mDeviceSurfaceVerticesIndexArray, mScalarFieldGridInfo.resolution, uint);

}

void ReconstructorGPUAkinci12::onFrameMove(unsigned int frameIndex)
{
	//ReconstructorGPU::onFrameMove(frameIndex);



}

void ReconstructorGPUAkinci12::onEndFrame(unsigned int frameIndex)
{
	//ReconstructorGPU::onEndFrame(frameIndex);

	CUDA_DESTROY_GRID(mDeviceIsSurfaceGrid);
	CUDA_DESTROY_GRID(mDeviceIsValidSurfaceGrid);
	CUDA_DESTROY_GRID(mDeviceSurfaceVerticesIndexArray);
}

void ReconstructorGPUAkinci12::onInitialization()
{
	//ReconstructorGPU::onInitialization();

}

void ReconstructorGPUAkinci12::onFinalization()
{
	//ReconstructorGPU::onFinalization();

}

void ReconstructorGPUAkinci12::saveFluidSurfaceObjToFile(unsigned int frameIndex)
{

}
