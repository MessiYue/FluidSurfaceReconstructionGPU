#include "ReconstructorGPU.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CudaUtils.h"
#include "ReconstructionCUDA.cuh"
#include "MarchingCubesHelper.h"

ReconstructorGPU::ReconstructorGPU(
	const std::string & directory,
	const std::string & filePattern,
	unsigned int from, unsigned int to) : 
	mFileDirectory(directory),
	mFilePattern(filePattern),
	mFrameFrom(from), mFrameTo(to),
	mSaveObjFile(true),
	mSaveVisFile(false),
	mSaveCfgFile(false) {}

void ReconstructorGPU::reconstruct()
{
	if (mFrameFrom >= mFrameTo)
	{
		std::cout << "Wrong settings occur for frame range.\n";
		return;
	}

	//! initialization.
	initialization();

	//! reconstructing frame by frame.
	for (auto frameIndex = mFrameFrom; frameIndex < mFrameTo; ++frameIndex)
	{
		std::cout << "\n*************************** Frame " << frameIndex << " ***************************\n";
		//! at the begining of each frame, do some preparation here.
		beginFrame(frameIndex);

		std::cout << "Number of particles: " << static_cast<double>(mNumParticles) / 1000.0 << " k.\n";

		std::cout << "Begin to reconstruc the fluid surface...\n";
		mTimer->reset();

		//! spatial hashing grid building.
		spatialHashingGridBuilding();

		//! reconstruct the surface.
		frameMove(frameIndex);

		double record = mTimer->durationInMilliseconds();
		mTimeConsuming.push_back(record);

		std::cout << "Reconstructing Frame" << frameIndex << " took " << record << " milliseconds.\n";

		//! save middle data to file for visualization.
		if (mSaveVisFile)
			saveMiddleDataToVisFile(frameIndex);

		//! get the configuration.
		if(mSaveCfgFile)
			getConfiguration(frameIndex);

		//! at the end of each frame, do some post-process here.
		endFrame(frameIndex);
	}

	//! save the time consumed to file.
	saveTimeConsumingRecordToFile();

	if (mSaveCfgFile)
		saveConfigurationToFile();

	//! finalization.
	finalization();
}

SimParam ReconstructorGPU::getSimulationParameters() const { return mSimParam; }

unsigned int ReconstructorGPU::getNumberOfParticles() const { return mNumParticles; }

std::vector<double> ReconstructorGPU::getTimeConsumingSequence() const { return mTimeConsuming; }

void ReconstructorGPU::setOutputMeshFile(bool flag) { mSaveObjFile = flag; }

void ReconstructorGPU::setOutputConfigFile(bool flag) { mSaveCfgFile = flag; }

void ReconstructorGPU::setOutputVisualizeFile(bool flag) { mSaveVisFile = flag; }

void ReconstructorGPU::beginFrame(unsigned int frameIndex)
{
	//! first of all, read the particles from file.
	std::vector<ParticlePosition> particles;
	std::vector<ScalarValue> densities;
	readParticlesFromFile(frameIndex, particles, densities);
	mScalarFieldGridInfo.cellSize = mSpatialGridInfo.cellSize / mSimParam.scSpGridResRatio;
	mScalarFieldGridInfo.minPos = mSpatialGridInfo.minPos;
	mScalarFieldGridInfo.maxPos = mSpatialGridInfo.maxPos;
	mScalarFieldGridInfo.resolution = make_uint3(
		(mScalarFieldGridInfo.maxPos.x - mScalarFieldGridInfo.minPos.x) / mScalarFieldGridInfo.cellSize + 1,
		(mScalarFieldGridInfo.maxPos.y - mScalarFieldGridInfo.minPos.y) / mScalarFieldGridInfo.cellSize + 1,
		(mScalarFieldGridInfo.maxPos.z - mScalarFieldGridInfo.minPos.z) / mScalarFieldGridInfo.cellSize + 1);

	std::cout << "Scalar Field Resolution: " << mScalarFieldGridInfo.resolution.x << " * "
		<< mScalarFieldGridInfo.resolution.y << " * " << mScalarFieldGridInfo.resolution.z << std::endl;

	//! copy the particles' positions and densities to gpu.
	mNumParticles = particles.size();
	CUDA_CREATE_GRID_1D(mDeviceParticlesArray, particles.size(), ParticlePosition);
	CUDA_CREATE_GRID_1D(mDeviceParticlesDensityArray, particles.size(), ScalarValue);
	CUDA_CREATE_GRID_1D(mDeviceSurfaceParticlesFlagGrid, particles.size(), uint);
	checkCudaErrors(cudaMemcpy(mDeviceParticlesArray.grid, static_cast<void*>(particles.data()),
		particles.size() * sizeof(ParticlePosition), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(mDeviceParticlesDensityArray.grid, static_cast<void*>(densities.data()),
		particles.size() * sizeof(ScalarValue), cudaMemcpyHostToDevice));

	//! memory allocation of other arrays.
	CUDA_CREATE_GRID_3D(mDeviceScalarFieldGrid, mScalarFieldGridInfo.resolution, ScalarValue);

	//! extra action for sub class.
	onBeginFrame(frameIndex);
}

void ReconstructorGPU::frameMove(unsigned int frameIndex)
{
	//! move on the frame.
	onFrameMove(frameIndex);
}

void ReconstructorGPU::endFrame(unsigned int frameIndex)
{
	//! first of all, save the mesh to file.
	if(mSaveObjFile)
		saveFluidSurfaceObjToFile(frameIndex);

	CUDA_DESTROY_GRID(mDeviceParticlesArray);
	CUDA_DESTROY_GRID(mDeviceParticlesDensityArray);
	CUDA_DESTROY_GRID(mDeviceSurfaceParticlesFlagGrid);
	CUDA_DESTROY_GRID(mDeviceScalarFieldGrid);
	CUDA_DESTROY_GRID(mDeviceCellParticleIndexArray);
	CUDA_DESTROY_GRID(mDeviceFlagGrid);

	//! extra action for sub class.
	onEndFrame(frameIndex);
}

void ReconstructorGPU::initialization()
{
	//! timer for recording.
	mTimer = std::shared_ptr<Timer>(new Timer());
	mStageTimer = std::shared_ptr<Timer>(new Timer());

	mSimParam.spatialCellSizeScale = 1.0;

	//! memory allocation for auxiliary textures.
	cudaMallocMemcpy((void**)&mDeviceEdgeTable, (void*)MarchingCubesHelper::edgeFlags, 256 * sizeof(uint));
	cudaMallocMemcpy((void**)&mDeviceEdgeIndicesOfTriangleTable, (void*)MarchingCubesHelper::edgeIndexesOfTriangle, 256 * 16 * sizeof(int));
	cudaMallocMemcpy((void**)&mDeviceNumVerticesTable, (void*)MarchingCubesHelper::numVertices, 256 * sizeof(uint));
	cudaMallocMemcpy((void**)&mDeviceVertexIndicesOfEdgeTable, (void*)MarchingCubesHelper::vertexIndexesOfEdge, 12 * 2 * sizeof(int));
	bindTextures(
		mDeviceEdgeTable,
		mDeviceEdgeIndicesOfTriangleTable,
		mDeviceNumVerticesTable,
		mDeviceVertexIndicesOfEdgeTable);

	//! initialization of arrays.
	INITGRID_ZERO(mDeviceParticlesArray);
	INITGRID_ZERO(mDeviceParticlesDensityArray);
	INITGRID_ZERO(mDeviceScalarFieldGrid);
	INITGRID_ZERO(mDeviceCellParticleIndexArray);
	INITGRID_ZERO(mDeviceFlagGrid);

	mTimeConsuming.clear();
	mSurfaceParRatio.clear();
	mSurfaceVerRatio.clear();
	mNumTriangles.clear();
	mStageTimeConsuming.clear();

	//! extra action for sub class.
	onInitialization();
}

void ReconstructorGPU::finalization()
{
	//! release of auxiliary textures.
	safeCudaFree((void**)&mDeviceEdgeTable);
	safeCudaFree((void**)&mDeviceEdgeIndicesOfTriangleTable);
	safeCudaFree((void**)&mDeviceNumVerticesTable);
	safeCudaFree((void**)&mDeviceVertexIndicesOfEdgeTable);



	//! extra action for sub class.
	onFinalization();
}

void ReconstructorGPU::readParticlesFromFile(
	unsigned int frameIndex,
	std::vector<ParticlePosition>& particles,
	std::vector<ScalarValue>& densities)
{
	char basename[256];
	snprintf(basename, sizeof(basename), mFilePattern.c_str(), frameIndex);
	std::string path = mFileDirectory + std::string(basename) + ".xyz";
	std::ifstream file(path.c_str());

	if (file)
	{
		std::cout << "Reading " << path << "...\n";
		std::string line;

		//! min point of bounding box.
		std::getline(file, line);
		std::stringstream ss1;
		ss1 << line;
		ss1 >> mSpatialGridInfo.minPos.x;
		ss1 >> mSpatialGridInfo.minPos.y;
		ss1 >> mSpatialGridInfo.minPos.z;

		//! max point of bounding box.
		std::getline(file, line);
		std::stringstream ss2;
		ss2 << line;
		ss2 >> mSpatialGridInfo.maxPos.x;
		ss2 >> mSpatialGridInfo.maxPos.y;
		ss2 >> mSpatialGridInfo.maxPos.z;

		//! kernel radius.
		std::getline(file, line);
		std::stringstream ss3;
		ss3 << line;
		ss3 >> mSimParam.smoothingRadius;

		mSimParam.smoothingRadiusInv = 1.0 / mSimParam.smoothingRadius;
		mSimParam.smoothingRadiusSq = mSimParam.smoothingRadius * mSimParam.smoothingRadius;
		mSpatialGridInfo.cellSize = mSimParam.smoothingRadius * mSimParam.spatialCellSizeScale;

		//! particle radius.
		std::getline(file, line);
		std::stringstream ss4;
		ss4 << line;
		ss4 >> mSimParam.particleRadius;

		//! particle mass.
		std::getline(file, line);
		std::stringstream ss5;
		ss5 << line;
		ss5 >> mSimParam.particleMass;

		//! expansion.
		mSpatialGridInfo.minPos.x -= mSpatialGridInfo.cellSize;
		mSpatialGridInfo.minPos.y -= mSpatialGridInfo.cellSize;
		mSpatialGridInfo.minPos.z -= mSpatialGridInfo.cellSize;
		mSpatialGridInfo.maxPos.x += mSpatialGridInfo.cellSize;
		mSpatialGridInfo.maxPos.y += mSpatialGridInfo.cellSize;
		mSpatialGridInfo.maxPos.z += mSpatialGridInfo.cellSize;

		//! calculation of resoluation.
		mSpatialGridInfo.resolution.x = (mSpatialGridInfo.maxPos.x - mSpatialGridInfo.minPos.x)
			/ mSpatialGridInfo.cellSize + 1;
		mSpatialGridInfo.resolution.y = (mSpatialGridInfo.maxPos.y - mSpatialGridInfo.minPos.y)
			/ mSpatialGridInfo.cellSize + 1;
		mSpatialGridInfo.resolution.z = (mSpatialGridInfo.maxPos.z - mSpatialGridInfo.minPos.z)
			/ mSpatialGridInfo.cellSize + 1;

		//! particles' positions and densities.
		while (std::getline(file, line))
		{
			std::stringstream str;
			str << line;
			ParticlePosition tmp;
			ScalarValue density;
			str >> tmp.pos.x;
			str >> tmp.pos.y;
			str >> tmp.pos.z;
			str >> density.value;
			particles.push_back(tmp);
			densities.push_back(density);
		}

		file.close();

		std::cout << "Finish reading " << path << ".\n";
	}
	else
		std::cout << "Failed to read the file:" << path << std::endl;
}

void ReconstructorGPU::spatialHashingGridBuilding()
{
	assert(
		mSpatialGridInfo.resolution.x > 0 &&
		mSpatialGridInfo.resolution.y > 0 &&
		mSpatialGridInfo.resolution.z > 0);

	//! memory allocation for virtual density grid and spatial hashing grid.
	CUDA_CREATE_GRID_3D(mDeviceFlagGrid, mSpatialGridInfo.resolution, float);
	CUDA_CREATE_GRID_3D(mDeviceCellParticleIndexArray, mSpatialGridInfo.resolution, IndexRange);
	cudaMemset(mDeviceFlagGrid.grid, 0.0f, mDeviceFlagGrid.size * sizeof(float));
	cudaMemset(mDeviceCellParticleIndexArray.grid, 0xffffffff, mDeviceCellParticleIndexArray.size * sizeof(IndexRange));

	//! launch the building kernel function.
	launchSpatialGridBuilding(
		mDeviceParticlesArray,
		mDeviceParticlesDensityArray,
		mNumParticles,
		mDeviceCellParticleIndexArray,
		mDeviceFlagGrid,
		mSpatialGridInfo);
}

void ReconstructorGPU::saveTimeConsumingRecordToFile()
{
	if (mTimeConsuming.empty())
		return;

	std::string fileName = getAlgorithmType();
	std::string path = mFileDirectory + fileName + ".time";
	std::ofstream file;
	file.open(path.c_str(), std::ios::out | std::ios::app);

	if (file)
	{
		std::cout << "Writing " << path << "...\n";
		for (auto frameIndex = mFrameFrom; frameIndex < mFrameTo; ++frameIndex)
		{
			double record = mTimeConsuming[frameIndex - mFrameFrom];
			file << record << ' ';
		}
		file << std::endl;
		file.close();
	}
	else
		std::cerr << "Failed to save the file: " << path << std::endl;
}

void ReconstructorGPU::saveConfigurationToFile()
{
	assert(mSurfaceParRatio.size() > 0 
		&& mSurfaceVerRatio.size() > 0 
		&& mNumTriangles.size() > 0);

	double avgSurParRatio = 0.0f;
	double avgSurVerRatio = 0.0f;
	int avgNumTriangles = 0;
	
	for (auto &elem : mSurfaceParRatio)
		avgSurParRatio += elem;
	avgSurParRatio /= mSurfaceParRatio.size();

	for (auto &elem : mSurfaceVerRatio)
		avgSurVerRatio += elem;
	avgSurVerRatio /= mSurfaceVerRatio.size();

	for (auto &elem : mNumTriangles)
		avgNumTriangles += elem;
	avgNumTriangles /= mNumTriangles.size();

	double avgEs = 0.0f;
	double avgSf = 0.0f;
	double avgTri = 0.0f;
	for (auto &elem : mStageTimeConsuming)
	{
		avgEs += elem.x;
		avgSf += elem.y;
		avgTri += elem.z;
	}
	avgEs /= mStageTimeConsuming.size();
	avgSf /= mStageTimeConsuming.size();
	avgTri /= mStageTimeConsuming.size();

	std::string fileName = getAlgorithmType();
	std::string path = mFileDirectory + fileName + ".config";
	std::ofstream file;
	file.open(path.c_str());
	if (file)
	{
		std::cout << "Writing " << path << "...\n";
		file << avgNumTriangles << std::endl;
		file << avgSurParRatio << std::endl;
		file << avgSurVerRatio << std::endl;
		file << avgEs << std::endl;
		file << avgEs + avgSf << std::endl;
		file << avgTri << std::endl;
		file << avgEs + avgSf + avgTri << std::endl;
		file.close();
	}
	else
		std::cerr << "Failed to save the file: " << path << std::endl;

}
