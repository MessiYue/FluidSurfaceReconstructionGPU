#include "ReconstructorGPUOursZB05.h"

#include "CudaUtils.h"
#include "ReconstructionCUDA.cuh"

ReconstructorGPUOursZB05::ReconstructorGPUOursZB05(const std::string & directory, const std::string & filePattern,
	unsigned int from, unsigned int to)
	: ReconstructorGPU(directory, filePattern, from, to) 
{
	onInitialization();
}

ReconstructorGPUOursZB05::~ReconstructorGPUOursZB05()
{
	onFinalization();
}

std::string ReconstructorGPUOursZB05::getAlgorithmType() { return std::string("Our Algorithm using ZB05 kernel"); }

void ReconstructorGPUOursZB05::onBeginFrame(unsigned int frameIndex)
{
	ReconstructorGPU::onBeginFrame(frameIndex);
}

void ReconstructorGPUOursZB05::onFrameMove(unsigned int frameIndex)
{
	//! step1: estimation of surface vertices.
	std::cout << "step1: estimation of surface vertices....\n";
	estimationOfSurfaceVertices();

	//! step2: compactation of surface vertices.
	std::cout << "step2: compactation of surface vertices...\n";
	compactationOfSurfaceVertices();

	//! step3: calculation of scalar field grid with compacted surface vertices.
	std::cout << "step3: calculation of scalar field grid...\n";
	computationOfScalarFieldGrid();

	//! step4: detection of valid surface cubes.
	std::cout << "step4: detection of valid surface cubes...\n";
	detectionOfValidSurfaceCubes();

	//! step5: compactation of valid surface cubes.
	std::cout << "step5: compactation of valid surface cubes...\n";
	compactationOfValidSurafceCubes();

	//! step6: generation of triangles for surface.
	std::cout << "step6: generation of triangles for surface...\n";
	generationOfSurfaceMeshUsingMC();

}

void ReconstructorGPUOursZB05::onEndFrame(unsigned int frameIndex)
{
	ReconstructorGPU::onEndFrame(frameIndex);
}

void ReconstructorGPUOursZB05::onInitialization()
{
	ReconstructorGPU::onInitialization();
	
	//! isocontour value.
	mSimParam.isoValue = -0.0001f;
	//! search extent.
	mSimParam.expandExtent = 3;
	//! 
	mSimParam.scSpGridResRatio = 2;
	//!
	mSimParam.spatialCellSizeScale = 1.0;
}

void ReconstructorGPUOursZB05::onFinalization()
{
	ReconstructorGPU::onFinalization();
	
}

void ReconstructorGPUOursZB05::estimationOfSurfaceVertices()
{
	//! calculation of grid dim and block dim for gpu threads.
	dim3 gridDim_, blockDim_;
	if (!calcGridDimBlockDim(mDeviceFlagGrid.size, gridDim_, blockDim_))
		return;

	//! set zero for mDeviceIsSurfaceGrid.
	checkCudaErrors(cudaMemset(mDeviceIsSurfaceGrid.grid, 0, mDeviceIsSurfaceGrid.size * sizeof(uint)));
	checkCudaErrors(cudaMemset(mDeviceIsSurfaceGridScan.grid, 0, mDeviceIsSurfaceGridScan.size * sizeof(uint)));

	//! launch the estimation kernel function.
	launchEstimateSurfaceVertices(gridDim_, blockDim_, mDeviceFlagGrid, mDeviceIsSurfaceGrid,
		mSimParam.expandExtent, mSimParam);
	getLastCudaError("launch estimationOfSurfaceVertices() failed");
}

void ReconstructorGPUOursZB05::compactationOfSurfaceVertices()
{
	//! calculation of exculsive prefix sum of mDeviceIsSurfaceGrid.
	mNumSurfaceVertices = launchThrustExclusivePrefixSumScan(mDeviceIsSurfaceGridScan.grid,
		mDeviceIsSurfaceGrid.grid, (uint)mDeviceIsSurfaceGrid.size);

	if (mNumSurfaceVertices <= 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}

	std::cout << "surface vertices ratio: " << static_cast<double>(mNumSurfaceVertices)
		/ (mScalarFieldGridInfo.resolution.x * mScalarFieldGridInfo.resolution.y * mScalarFieldGridInfo.resolution.z) << std::endl;

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mDeviceIsSurfaceGrid.size, gridDim_, blockDim_);

	//! launch the compactation kernel function.
	launchCompactSurfaceVertex(gridDim_, blockDim_, mDeviceSurfaceVerticesIndexArray, mDeviceIsSurfaceGridScan,
		mDeviceIsSurfaceGrid, mSimParam);
	getLastCudaError("launch compactationOfSurfaceVertices() failed");
}

void ReconstructorGPUOursZB05::computationOfScalarFieldGrid()
{
	if (mNumSurfaceVertices <= 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumSurfaceVertices, gridDim_, blockDim_);

	//! set zero for scalar field grid.
	checkCudaErrors(cudaMemset(mDeviceScalarFieldGrid.grid, 0, mDeviceScalarFieldGrid.size * sizeof(ScalarValue)));

	//! launch the computation kernel function.
	launchUpdateScalarGridValuesCompacted(gridDim_, blockDim_, mDeviceSurfaceVerticesIndexArray,
		mNumSurfaceVertices, mDeviceCellParticleIndexArray, mDeviceParticlesArray, mDeviceScalarFieldGrid,
		mSpatialGridInfo, mScalarFieldGridInfo, mSimParam);
	getLastCudaError("launch computationOfScalarFieldGrid() failed");
}
