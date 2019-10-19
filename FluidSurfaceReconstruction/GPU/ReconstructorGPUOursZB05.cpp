#include "ReconstructorGPUOursZB05.h"

#include "CudaUtils.h"
#include "ReconstructionCUDA.cuh"

ReconstructorGPUOursZB05::ReconstructorGPUOursZB05(const std::string & directory, const std::string & filePattern,
	unsigned int from, unsigned int to) : ReconstructorGPU(directory, filePattern, from, to) 
{
}

ReconstructorGPUOursZB05::~ReconstructorGPUOursZB05() {}

std::string ReconstructorGPUOursZB05::getAlgorithmType() { return std::string("Our Algorithm using ZB05 kernel"); }

void ReconstructorGPUOursZB05::onBeginFrame(unsigned int frameIndex)
{
	//! nothing to do.
}

void ReconstructorGPUOursZB05::onFrameMove(unsigned int frameIndex)
{
	//! step1: extraction of surface particles.
	std::cout << "step1: extraction of surface particles....\n";
	extractionOfSurfaceParticles();

	//! step2: estimation of surface vertices.
	std::cout << "step2: estimation of surface vertices....\n";
	estimationOfSurfaceVertices();

	//! step2: compactation of surface vertices.
	std::cout << "step3: compactation of surface vertices...\n";
	compactationOfSurfaceVertices();

	//! step3: calculation of scalar field grid with compacted surface vertices.
	std::cout << "step4: calculation of scalar field grid...\n";
	computationOfScalarFieldGrid();

	//! step4: detection of valid surface cubes.
	std::cout << "step5: detection of valid surface cubes...\n";
	detectionOfValidSurfaceCubes();

	//! step5: compactation of valid surface cubes.
	std::cout << "step6: compactation of valid surface cubes...\n";
	compactationOfValidSurafceCubes();

	//! step6: generation of triangles for surface.
	std::cout << "step7: generation of triangles for surface...\n";
	generationOfSurfaceMeshUsingMC();

}

void ReconstructorGPUOursZB05::onEndFrame(unsigned int frameIndex) 
{
	//! nothing to do.
}

void ReconstructorGPUOursZB05::onInitialization()
{	
	//! isocontour value.
	mSimParam.isoValue = -0.0001f;
	//! search extent.
	mSimParam.expandExtent = 4;
	//! 
	mSimParam.scSpGridResRatio = 4;
	//!
	mSimParam.spatialCellSizeScale = 1.0;
}

void ReconstructorGPUOursZB05::onFinalization() 
{
	//! nothing to do.
}

void ReconstructorGPUOursZB05::saveMiddleDataToVisFile(unsigned int frameIndex)
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
		if(flagArray.size() > 0)
			file << std::endl;
		std::vector<float>().swap(flagArray);

		//! smoothed particles, none.
		file << 0 << std::endl;

		//! surface vertices' indcies of scalar field grid.
		file << mNumSurfaceVertices << std::endl;
		std::vector<uint> surfaceVerticesIndexArray;
		surfaceVerticesIndexArray.resize(mNumSurfaceVertices);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(surfaceVerticesIndexArray.data()),
			mDeviceSurfaceVerticesIndexArray.grid, sizeof(uint) * mNumSurfaceVertices, cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < mNumSurfaceVertices; ++i)
			file << surfaceVerticesIndexArray[i] << ' ';
		if(mNumSurfaceVertices > 0)
			file << std::endl;
		std::vector<uint>().swap(surfaceVerticesIndexArray);

		//! surface particles, none.
		std::vector<uint> surfaceParticlesFlagArray;
		surfaceParticlesFlagArray.resize(mNumParticles);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(surfaceParticlesFlagArray.data()),
			mDeviceSurfaceParticlesFlagGrid.grid, sizeof(uint) * mNumParticles, cudaMemcpyDeviceToHost));
		uint numSurfaceParticles = 0;
		for (size_t i = 0; i < surfaceParticlesFlagArray.size(); ++i)
			if (surfaceParticlesFlagArray[i] == 1)
				++numSurfaceParticles;
		file << numSurfaceParticles << std::endl;
		for (size_t i = 0; i < surfaceParticlesFlagArray.size(); ++i)
		{
			if (surfaceParticlesFlagArray[i] == 1)
				file << i << ' ';
		}
		if (numSurfaceParticles > 0)
			file << std::endl;
		std::vector<uint>().swap(surfaceParticlesFlagArray);

		//! involve particles, none.
		file << 0 << std::endl;

		//! valid surface cubes.
		file << mNumValidSurfaceCubes << std::endl;
		std::vector<uint> validCubesIndexArray;
		validCubesIndexArray.resize(mNumValidSurfaceCubes);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(validCubesIndexArray.data()),
			mDeviceValidSurfaceIndexArray.grid, sizeof(uint) * mDeviceValidSurfaceIndexArray.size, cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < mNumValidSurfaceCubes; ++i)
			file << validCubesIndexArray[i] << ' ';
		if(mNumValidSurfaceCubes > 0)
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

void ReconstructorGPUOursZB05::extractionOfSurfaceParticles()
{
	//! calculation of grid dim and block dim for gpu threads.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mDeviceCellParticleIndexArray.size, gridDim_, blockDim_);

	//! initialization for surface particle flag.
	checkCudaErrors(cudaMemset(mDeviceSurfaceParticlesFlagGrid.grid, 0,
		mNumParticles * sizeof(uint)));

	//! launch extraction kernel function.
	launchExtractionOfSurfaceParticles(
		gridDim_,
		blockDim_,
		mSimParam,
		mDeviceFlagGrid,
		mDeviceSurfaceParticlesFlagGrid,
		mDeviceCellParticleIndexArray);
	getLastCudaError("launch extractionOfSurfaceParticles() failed");
}

void ReconstructorGPUOursZB05::estimationOfSurfaceVertices()
{
	//! calculation of grid dim and block dim for gpu threads.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumParticles, gridDim_, blockDim_);

	//! set zero for mDeviceIsSurfaceGrid.
	checkCudaErrors(cudaMemset(mDeviceIsSurfaceGrid.grid, 0, mDeviceIsSurfaceGrid.size * sizeof(uint)));

	//! launch the estimation kernel function.
	launchEstimationOfSurfaceVertices(
		gridDim_,
		blockDim_,
		mSimParam,
		mScalarFieldGridInfo,
		mDeviceParticlesArray,
		mDeviceSurfaceParticlesFlagGrid,
		mDeviceCellParticleIndexArray,
		mDeviceIsSurfaceGrid);
	getLastCudaError("launch estimationOfSurfaceVertices() failed");
}

void ReconstructorGPUOursZB05::compactationOfSurfaceVertices()
{
	//! calculation of exculsive prefix sum of mDeviceIsSurfaceGrid.
	mNumSurfaceVertices = launchThrustExclusivePrefixSumScan(mDeviceIsSurfaceGridScan.grid,
		mDeviceIsSurfaceGrid.grid, (uint)mDeviceIsSurfaceGrid.size);

	if (mNumSurfaceVertices == 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}

	std::cout << "surface vertices ratio: " << static_cast<double>(mNumSurfaceVertices) / 
		(mScalarFieldGridInfo.resolution.x * mScalarFieldGridInfo.resolution.y
			* mScalarFieldGridInfo.resolution.z) << std::endl;

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mDeviceIsSurfaceGrid.size, gridDim_, blockDim_);

	//! launch the compactation kernel function.
	launchCompactSurfaceVertex(
		gridDim_,
		blockDim_,
		mDeviceSurfaceVerticesIndexArray,
		mDeviceIsSurfaceGridScan,
		mDeviceIsSurfaceGrid,
		mSimParam);
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
	launchUpdateScalarGridValuesCompacted(
		gridDim_,
		blockDim_,
		mDeviceSurfaceVerticesIndexArray,
		mNumSurfaceVertices,
		mDeviceCellParticleIndexArray,
		mDeviceParticlesArray,
		mDeviceScalarFieldGrid,
		mSpatialGridInfo,
		mScalarFieldGridInfo,
		mSimParam);
	getLastCudaError("launch computationOfScalarFieldGrid() failed");
}
