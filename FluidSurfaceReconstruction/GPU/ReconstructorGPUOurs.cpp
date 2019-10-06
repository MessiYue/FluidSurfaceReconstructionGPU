#include "ReconstructorGPUOurs.h"

#include "CudaUtils.h"
#include "SurfaceReconstructionCUDA.cuh"

ReconstructorGPUOurs::ReconstructorGPUOurs(const std::string & directory, const std::string & filePattern, 
	unsigned int from, unsigned int to)
	: ReconstructorGPU(directory, filePattern, from, to) {}

ReconstructorGPUOurs::~ReconstructorGPUOurs() {}

std::string ReconstructorGPUOurs::getAlgorithmType() { return std::string("Our Algorithm"); }

void ReconstructorGPUOurs::onBeginFrame(unsigned int frameIndex)
{
	ReconstructorGPU::onBeginFrame(frameIndex);
}

void ReconstructorGPUOurs::onFrameMove(unsigned int frameIndex)
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

void ReconstructorGPUOurs::onEndFrame(unsigned int frameIndex)
{
	ReconstructorGPU::onEndFrame(frameIndex);
}

void ReconstructorGPUOurs::estimationOfSurfaceVertices()
{
	//! calculation of grid dim and block dim for gpu threads.
	dim3 gridDim_, blockDim_;
	if (!calcGridDimBlockDim(mDeviceDensityGrid.size, gridDim_, blockDim_))
		return;

	//! set zero for mDeviceIsSurfaceGrid.
	checkCudaErrors(cudaMemset(mDeviceIsSurfaceGrid.grid, 0, mDeviceIsSurfaceGrid.size * sizeof(uint)));

	//! launch the estimation kernel function.
	launchEstimateSurfaceVertices(gridDim_, blockDim_, mDeviceDensityGrid, mDeviceIsSurfaceGrid,
		mSimParam.searchExtent, mSimParam);
	getLastCudaError("launch estimationOfSurfaceVertices() failed");
}

void ReconstructorGPUOurs::compactationOfSurfaceVertices()
{
	//! calculation of exculsive prefix sum of mDeviceIsSurfaceGrid.
	mNumSurfaceVertices = ThrustExclusiveScanWrapper(mDeviceIsSurfaceGridScan.grid,
		mDeviceIsSurfaceGrid.grid, (uint)mDeviceIsSurfaceGrid.size);

	if (mNumSurfaceVertices <= 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mDeviceIsSurfaceGrid.size, gridDim_, blockDim_);

	//! launch the compactation kernel function.
	launchCompactSurfaceVertex(gridDim_, blockDim_, mDeviceSurfaceVertexIndexArray, mDeviceIsSurfaceGridScan,
		mDeviceIsSurfaceGrid, mSimParam);
	getLastCudaError("launch compactationOfSurfaceVertices() failed");
}

void ReconstructorGPUOurs::computationOfScalarFieldGrid()
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
	checkCudaErrors(cudaMemset(mDeviceScalarFieldGrid.grid, 0, mDeviceScalarFieldGrid.size * sizeof(SimpleVertex)));

	//! launch the computation kernel function.
	launchUpdateScalarGridValuesCompacted(gridDim_, blockDim_, mDeviceSurfaceVertexIndexArray,
		mNumSurfaceVertices, mDeviceCellParticleIndexArray, mDeviceParticlesArray, mDeviceScalarFieldGrid,
		mSpatialGridInfo, mScalarFieldGridInfo, mSimParam);
	getLastCudaError("launch computationOfScalarFieldGrid() failed");
}

void ReconstructorGPUOurs::detectionOfValidSurfaceCubes()
{
	if (mNumSurfaceVertices <= 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}

	//! memory allocation for detection of valid surface cubes.
	CUDA_CREATE_GRID_1D_SET(mDeviceIsValidSurfaceGrid, mNumSurfaceVertices, mNumSurfaceVertices, 0, uint);
	CUDA_CREATE_GRID_1D_SET(mDeviceIsValidSurfaceGridScan, mNumSurfaceVertices, mNumSurfaceVertices, 0, uint);
	CUDA_CREATE_GRID_1D_SET(mDeviceNumVerticesGrid, mNumSurfaceVertices, mNumSurfaceVertices, 0, uint);
	CUDA_CREATE_GRID_1D_SET(mDeviceNumVerticesGridScan, mNumSurfaceVertices, mNumSurfaceVertices, 0, uint);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumSurfaceVertices, gridDim_, blockDim_);

	//! launch the detection kernel function.
	launchDetectValidSurfaceCubes(gridDim_, blockDim_, mDeviceSurfaceVertexIndexArray, mNumSurfaceVertices,
		mDeviceScalarFieldGrid, mDeviceIsValidSurfaceGrid, mDeviceNumVerticesGrid, mDeviceIsSurfaceGrid, mSimParam);
}

void ReconstructorGPUOurs::compactationOfValidSurafceCubes()
{
	//! calculation of exclusive prefix sum of mDeviceIsValidSurfaceGrid.
	mNumValidSurfaceCubes = ThrustExclusiveScanWrapper(mDeviceIsValidSurfaceGridScan.grid,
		mDeviceIsValidSurfaceGrid.grid, (uint)mDeviceIsValidSurfaceGrid.size);

	//! calculation of exclusive prefix sum of mDeviceNumVerticesGrid.
	mNumSurfaceMeshVertices = ThrustExclusiveScanWrapper(mDeviceNumVerticesGridScan.grid,
		mDeviceNumVerticesGrid.grid, (uint)mDeviceNumVerticesGrid.size);

	if (mNumSurfaceMeshVertices <= 0)
	{
		std::cerr << "No vertex of surface mesh detected!\n";
		return;
	}

	//£¡memory allocation of valid surface cubes.
	SAFE_CUDA_FREE_GRID(mDeviceValidSurfaceIndexArray);
	CUDA_CREATE_GRID_1D_SET(mDeviceValidSurfaceIndexArray, mNumValidSurfaceCubes, mNumValidSurfaceCubes, 0, uint);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumSurfaceVertices, gridDim_, blockDim_);

	//! launch the compactation kernel function.
	launchCompactValidSurfaceCubes(gridDim_, blockDim_, mDeviceValidSurfaceIndexArray,
		mDeviceIsValidSurfaceGridScan, mDeviceIsValidSurfaceGrid, mSimParam);

}

void ReconstructorGPUOurs::generationOfSurfaceMeshUsingMC()
{
	if (mNumSurfaceMeshVertices <= 0)
	{
		std::cerr << "No vertex of surface mesh detected!\n";
		return;
	}

	//! memory allocation for generation of triangles for surface.
	CUDA_CREATE_GRID_1D_SET(mDeviceVertexArray, mNumSurfaceMeshVertices, mNumSurfaceMeshVertices, 0, float3);
	CUDA_CREATE_GRID_1D_SET(mDeviceNormalArray, mNumSurfaceMeshVertices, mNumSurfaceMeshVertices, 0, float3);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumValidSurfaceCubes, gridDim_, blockDim_);

	//! launch the generation kernel function.
	launchGenerateTriangles(gridDim_, blockDim_, mDeviceSurfaceVertexIndexArray, mDeviceValidSurfaceIndexArray,
		mScalarFieldGridInfo, mDeviceNumVerticesGridScan, mDeviceScalarFieldGrid, mDeviceVertexArray,
		mDeviceNormalArray, mSimParam);
}
