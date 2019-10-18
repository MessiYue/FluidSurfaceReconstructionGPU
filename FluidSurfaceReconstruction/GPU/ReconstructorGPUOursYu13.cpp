#include "ReconstructorGPUOursYu13.h"

#include "ReconstructionCUDA.cuh"

ReconstructorGPUOurYu13::ReconstructorGPUOurYu13(const std::string & directory, const std::string & filePattern,
	unsigned int from, unsigned int to) : ReconstructorGPU(directory, filePattern, from ,to) {}

ReconstructorGPUOurYu13::~ReconstructorGPUOurYu13() {}

std::string ReconstructorGPUOurYu13::getAlgorithmType() { return std::string("Our Algorithm using anisotropic kernel"); }

void ReconstructorGPUOurYu13::onBeginFrame(unsigned int frameIndex)
{
	//! memory allocation for extra data storage.
	CUDA_CREATE_GRID_3D(mDeviceNumSurfaceParticlesGrid, mSpatialGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(mDeviceNumSurfaceParticlesGridScan, mSpatialGridInfo.resolution, uint);
	CUDA_CREATE_GRID_1D(mDeviceParticlesMean, mDeviceParticlesArray.size, ParticlePosition);
	CUDA_CREATE_GRID_1D(mDeviceParticlesSmoothed, mDeviceParticlesArray.size, ParticlePosition);

	//! r = 2h.
	mSimParam.anisotropicRadius = mSimParam.smoothingRadius * 2;
}

void ReconstructorGPUOurYu13::onFrameMove(unsigned int frameIndex)
{
	//! step1: calculation of mean and smoothed positions of particles.
	std::cout << "step1: calculation of mean and smoothed positions of particles....\n";
	calculationOfMeanAndSmoothedParticles();

	//! step2: estimation of surface vertices and surface particles.
	std::cout << "step2: estimation of surface vertices and surface particles....\n";
	estimationOfSurfaceVerticesAndParticles();

	//! step3: compactation of surface vertices and surface particles.
	std::cout << "step3: compactation of surface particles....\n";
	compactationOfSurfaceVerticesAndParticles();

	//! step4: calculation of transform matrices for each surface particle.
	std::cout << "step4: calculation of transform matrices for each surface particle....\n";
	calculationOfTransformMatricesForParticles();

	//! step5: calculation of scalar field grid with compacted surface vertices and surface particles.
	std::cout << "step5: calculation of scalar field grid....\n";
	computationOfScalarFieldGrid();

	//! step6: detection of valid surface cubes.
	std::cout << "step6: detection of valid surface cubes...\n";
	detectionOfValidSurfaceCubes();

	//! step7: compactation of valid surface cubes.
	std::cout << "step7: compactation of valid surface cubes...\n";
	compactationOfValidSurafceCubes();

	//! step8: generation of triangles for surface.
	std::cout << "step8: generation of triangles for surface...\n";
	generationOfSurfaceMeshUsingMC();

}

void ReconstructorGPUOurYu13::onEndFrame(unsigned int frameIndex)
{
	CUDA_DESTROY_GRID(mDeviceNumSurfaceParticlesGrid);
	CUDA_DESTROY_GRID(mDeviceNumSurfaceParticlesGridScan);
	CUDA_DESTROY_GRID(mDeviceParticlesMean);
	CUDA_DESTROY_GRID(mDeviceParticlesSmoothed);
	CUDA_DESTROY_GRID(mDeviceSurfaceParticlesIndexArray);
	CUDA_DESTROY_GRID(mDeviceSVDMatricesArray);
}

void ReconstructorGPUOurYu13::onInitialization()
{
	//! isocontour value.
	mSimParam.isoValue = 0.0f;
	//! search extent.
	mSimParam.expandExtent = 2;
	//! 
	mSimParam.scSpGridResRatio = 2;
	//! appropriate scaling for spatial hashing grid.
	mSimParam.spatialCellSizeScale = 1.3f;
	//! blending parameter for particle smoothing.
	mSimParam.lambdaForSmoothed = 0.5f;
	//! minimal number of neighbors for non isolated particle.
	mSimParam.minNumNeighbors = 25;

	INITGRID_ZERO(mDeviceParticlesMean);
	INITGRID_ZERO(mDeviceParticlesSmoothed);
	INITGRID_ZERO(mDeviceNumSurfaceParticlesGrid);
	INITGRID_ZERO(mDeviceSurfaceParticlesIndexArray);
	INITGRID_ZERO(mDeviceSVDMatricesArray);
}

void ReconstructorGPUOurYu13::onFinalization()
{
	//! nothing to do.
}

void ReconstructorGPUOurYu13::saveMiddleDataToVisFile(unsigned int frameIndex)
{
	//! save middle data for visualization.
	char basename[256];
	snprintf(basename, sizeof(basename), mFilePattern.c_str(), frameIndex);
	std::string path = mFileDirectory + std::string(basename) + ".vis";
	std::ofstream file(path.c_str());
	if (file)
	{
		std::cout << "Writing to " << path << "...\n";

		//! spatial hashing grid bounding box.
		file << mSpatialGridInfo.minPos.x << ' ' << mSpatialGridInfo.minPos.y << ' '
			<< mSpatialGridInfo.minPos.z << std::endl;
		file << mSpatialGridInfo.maxPos.x << ' ' << mSpatialGridInfo.maxPos.y << ' '
			<< mSpatialGridInfo.maxPos.z << std::endl;
		file << mSpatialGridInfo.resolution.x << ' ' << mSpatialGridInfo.resolution.y << ' '
			<< mSpatialGridInfo.resolution.z << std::endl;
		file << mSpatialGridInfo.cellSize << std::endl;

		//! scalar field grid bounding box.
		file << mScalarFieldGridInfo.minPos.x << ' ' << mScalarFieldGridInfo.minPos.y
			<< ' ' << mScalarFieldGridInfo.minPos.z << std::endl;
		file << mScalarFieldGridInfo.maxPos.x << ' ' << mScalarFieldGridInfo.maxPos.y
			<< ' ' << mScalarFieldGridInfo.maxPos.z << std::endl;
		file << mScalarFieldGridInfo.resolution.x << ' ' << mScalarFieldGridInfo.resolution.y << ' '
			<< mScalarFieldGridInfo.resolution.z << std::endl;
		file << mScalarFieldGridInfo.cellSize << std::endl;

		//! particles .xyz file path.
		file << (std::string(basename) + ".xyz") << std::endl;

		//! flag of spatial hashing grid.
		std::vector<float> flagArray;
		flagArray.resize(mDeviceFlagGrid.size);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(flagArray.data()), mDeviceFlagGrid.grid,
			sizeof(float) * mDeviceFlagGrid.size, cudaMemcpyDeviceToHost));
		unsigned int numOfValidFlag = 0;
		for (size_t i = 0; i < flagArray.size(); ++i)
		{
			if (flagArray[i] > 0.0f)
				++numOfValidFlag;
		}
		file << numOfValidFlag << std::endl;
		for (size_t i = 0; i < flagArray.size(); ++i)
		{
			if (flagArray[i] > 0.0f)
				file << i << ' ';
		}
		if (flagArray.size() > 0)
			file << std::endl;
		std::vector<float>().swap(flagArray);

		//! smoothed particles.
		file << mNumParticles << std::endl;
		std::vector<ParticlePosition> smoothedParticles;
		smoothedParticles.resize(mNumParticles);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(smoothedParticles.data()),
			mDeviceParticlesSmoothed.grid, sizeof(ParticlePosition) * mNumParticles, cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < mNumParticles; ++i)
		{
			file << smoothedParticles[i].pos.x << ' ' << smoothedParticles[i].pos.y << ' '
				<< smoothedParticles[i].pos.z << std::endl;
		}
		std::vector<ParticlePosition>().swap(smoothedParticles);

		//! surface vertices' indcies of scalar field grid.
		file << mNumSurfaceVertices << std::endl;
		std::vector<uint> surfaceVerticesIndexArray;
		surfaceVerticesIndexArray.resize(mNumSurfaceVertices);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(surfaceVerticesIndexArray.data()),
			mDeviceSurfaceVerticesIndexArray.grid, sizeof(uint) * mNumSurfaceVertices, cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < mNumSurfaceVertices; ++i)
			file << surfaceVerticesIndexArray[i] << ' ';
		if (mNumSurfaceVertices > 0)
			file << std::endl;
		std::vector<uint>().swap(surfaceVerticesIndexArray);

		//! surface particles.
		file << mNumSurfaceParticles << std::endl;
		std::vector<uint> surfaceParticlesIndexArray;
		surfaceParticlesIndexArray.resize(mNumSurfaceParticles);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(surfaceParticlesIndexArray.data()),
			mDeviceSurfaceParticlesIndexArray.grid, sizeof(uint) * mNumSurfaceParticles, cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < mNumSurfaceParticles; ++i)
			file << surfaceParticlesIndexArray[i] << ' ';
		if (mNumSurfaceParticles > 0)
			file << std::endl;
		std::vector<uint>().swap(surfaceParticlesIndexArray);

		//! valid surface cubes.
		file << mNumValidSurfaceCubes << std::endl;
		std::vector<uint> validCubesIndexArray;
		validCubesIndexArray.resize(mNumValidSurfaceCubes);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(validCubesIndexArray.data()),
			mDeviceValidSurfaceIndexArray.grid, sizeof(uint) * mDeviceValidSurfaceIndexArray.size, cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < mNumValidSurfaceCubes; ++i)
			file << validCubesIndexArray[i] << ' ';
		if (mNumValidSurfaceCubes > 0)
			file << std::endl;

		//! surface mesh file.
		file << (std::string(basename) + ".obj") << std::endl;

		//! neighbourhood extent radius.
		file << mSimParam.smoothingRadius << std::endl;

		file.close();

		std::cout << "Finish writing " << path << ".\n";
	}
	else
		std::cerr << "Failed to save the file: " << path << std::endl;
}

void ReconstructorGPUOurYu13::calculationOfMeanAndSmoothedParticles()
{
	//! calculation of grid dim and block dim for gpu threads.
	dim3 gridDim_, blockDim_;
	if (!calcGridDimBlockDim(mDeviceParticlesArray.size, gridDim_, blockDim_))
		return;
	
	//! launch the calculation kernel function.
	launchCalculationOfMeanAndSmoothParticles(gridDim_, blockDim_, mSimParam, mDeviceParticlesArray,
		mDeviceParticlesMean, mDeviceParticlesSmoothed, mDeviceCellParticleIndexArray, mSpatialGridInfo);
	getLastCudaError("launch estimationOfSurfaceVertices() failed");

}

void ReconstructorGPUOurYu13::estimationOfSurfaceVerticesAndParticles()
{
	//! calculation of grid dim and block dim for gpu threads.
	dim3 gridDim_, blockDim_;
	if (!calcGridDimBlockDim(mDeviceFlagGrid.size, gridDim_, blockDim_))
		return;

	//! set zero for mDeviceNumSurfaceParticlesGrid.
	checkCudaErrors(cudaMemset(mDeviceNumSurfaceParticlesGrid.grid, 0, mDeviceNumSurfaceParticlesGrid.size * sizeof(uint)));
	
	//! launch the estimation kernel function.
	launchEstimationOfSurfaceVerticesAndParticles(
		gridDim_,
		blockDim_,
		mSimParam,
		mDeviceFlagGrid,
		mDeviceIsSurfaceGrid,
		mDeviceNumSurfaceParticlesGrid,
		mDeviceCellParticleIndexArray,
		mSpatialGridInfo);
	getLastCudaError("launch estimationOfSurfaceVerticesAndParticles() failed");

}

void ReconstructorGPUOurYu13::compactationOfSurfaceVerticesAndParticles()
{
	//! calculation of exclusive prefix sum of mDeviceNumSurfaceParticlesGrid.
	mNumSurfaceParticles = launchThrustExclusivePrefixSumScan(
		mDeviceNumSurfaceParticlesGridScan.grid,
		mDeviceNumSurfaceParticlesGrid.grid, 
		(uint)mDeviceNumSurfaceParticlesGrid.size);
	mNumSurfaceVertices = launchThrustExclusivePrefixSumScan(
		mDeviceIsSurfaceGridScan.grid,
		mDeviceIsSurfaceGrid.grid,
		(uint)mDeviceIsSurfaceGrid.size);

	if (mNumSurfaceParticles == 0)
	{
		std::cerr << "No surface particle detected!\n";
		return;
	}

	if (mNumSurfaceVertices == 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}

	std::cout << "surface vertices ratio: " << static_cast<double>(mNumSurfaceVertices) /
		(mScalarFieldGridInfo.resolution.x * mScalarFieldGridInfo.resolution.y
			* mScalarFieldGridInfo.resolution.z) << std::endl;
	std::cout << "surface particles ratio: " << static_cast<double>(mNumSurfaceParticles)
		/ mNumParticles << std::endl;

	//! memory allocation for surface particles.
	CUDA_CREATE_GRID_1D_SET(mDeviceSurfaceParticlesIndexArray, mNumSurfaceParticles,
		mNumSurfaceParticles, 0, uint);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	auto totalThreads = mDeviceNumSurfaceParticlesGrid.size;
	if (totalThreads < mDeviceIsSurfaceGrid.size)
		totalThreads = mDeviceIsSurfaceGrid.size;
	calcGridDimBlockDim(totalThreads, gridDim_, blockDim_);

	//! launch the compactation of surface particles.
	launchCompactationOfSurfaceVerticesAndParticles(
		gridDim_,
		blockDim_,
		mSimParam,
		mDeviceIsSurfaceGrid,
		mDeviceIsSurfaceGridScan,
		mDeviceNumSurfaceParticlesGrid,
		mDeviceNumSurfaceParticlesGridScan,
		mDeviceCellParticleIndexArray,
		mDeviceSurfaceVerticesIndexArray,
		mDeviceSurfaceParticlesIndexArray);
	getLastCudaError("launch compactationOfSurfaceVerticesAndParticles() failed");

}

void ReconstructorGPUOurYu13::calculationOfTransformMatricesForParticles()
{
	//! memory allocation for surface particles' matrices.
	CUDA_CREATE_GRID_1D_SET(mDeviceSVDMatricesArray, mNumSurfaceParticles,
		mNumSurfaceParticles, 0, MatrixValue);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumSurfaceParticles, gridDim_, blockDim_);

	//! launch the calculation kernel function.
	launchCalculationOfTransformMatricesForParticles(
		gridDim_,
		blockDim_, 
		mSimParam,
		mSpatialGridInfo,
		mDeviceParticlesMean,
		mDeviceParticlesArray,
		mDeviceCellParticleIndexArray,
		mDeviceSurfaceParticlesIndexArray,
		mDeviceSVDMatricesArray);
	getLastCudaError("launch calculationOfTransformMatricesForParticles() failed");
}

void ReconstructorGPUOurYu13::computationOfScalarFieldGrid()
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
	checkCudaErrors(cudaMemset(mDeviceScalarFieldGrid.grid, 100, mDeviceScalarFieldGrid.size * sizeof(ScalarValue)));

	//! launch the computation kernel function.
	launchComputationOfScalarFieldGrid(
		gridDim_,
		blockDim_,
		mSimParam,
		mNumSurfaceVertices,
		mScalarFieldGridInfo,
		mSpatialGridInfo,
		mDeviceSVDMatricesArray,
		mDeviceParticlesSmoothed,
		mDeviceParticlesDensityArray,
		mDeviceCellParticleIndexArray,
		mDeviceSurfaceVerticesIndexArray,
		mDeviceNumSurfaceParticlesGrid,
		mDeviceNumSurfaceParticlesGridScan,
		mDeviceScalarFieldGrid);
	getLastCudaError("launch computationOfScalarFieldGrid() failed");
}
