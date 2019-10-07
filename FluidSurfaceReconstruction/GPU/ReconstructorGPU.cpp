#include "ReconstructorGPU.h"

#include <iostream>
#include <fstream>
#include <istream>
#include <assert.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_runtime_api.h>

#include "Kernel.cuh"
#include "CudaUtils.h"
#include "SurfaceReconstructionCUDA.cuh"
#include "../CPU/MarchingCubesHelper.h"
#include "../CPU/MathUtils.h"

ReconstructorGPU::ReconstructorGPU(const std::string & directory, const std::string & filePattern,
	unsigned int from, unsigned int to) : 
	mFileDirectory(directory), mFilePattern(filePattern),
	mFrameFrom(from), mFrameTo(to)
{ 
	onInitialization();
}

ReconstructorGPU::~ReconstructorGPU() { onFinalization(); }

void ReconstructorGPU::reconstruct()
{
	if (mFrameFrom >= mFrameTo)
	{
		std::cout << "Wrong settings occur for frame range.\n";
		return;
	}

	mTimeConsuming.clear();

	//! reconstructing frame by frame.
	for (auto frameIndex = mFrameFrom; frameIndex < mFrameTo; ++frameIndex)
	{
		std::cout << "\n*************************** Frame " << frameIndex << " ***************************\n";
		//! at the begining of each frame, do some preparation here.
		onBeginFrame(frameIndex);

		std::cout << "Number of particles: " << static_cast<double>(mNumParticles) / 1000.0 << " k.\n";

		std::cout << "Begin to reconstruc the fluid surface...\n";
		double record = 0.0;
		mTimer->reset();

		//! reconstruct the surface.
		onFrameMove(frameIndex);

		record = mTimer->durationInMilliseconds();
		mTimeConsuming.push_back(record);
		
		std::cout << "Reconstructing Frame" << frameIndex << " took " << record << " milliseconds.\n";

		//! at the end of each frame, do some post-process here.
		onEndFrame(frameIndex);
	}

	//! save the time consumed to file.
	saveTimeConsumingRecordToFile();
}

SimParam ReconstructorGPU::getSimulationParameters() const { return mSimParam; }

unsigned int ReconstructorGPU::getNumberOfParticles() const { return mNumParticles; }

std::vector<double> ReconstructorGPU::getTimeConsumingSequence() const { return mTimeConsuming; }

void ReconstructorGPU::onBeginFrame(unsigned int frameIndex)
{
	//! first of all, read the particles from file.
	std::vector<SimpleParticle> particles = readParticlesFromFile(frameIndex);
	mScalarFieldGridInfo.cellSize = mSpatialGridInfo.cellSize * 0.5;
	mScalarFieldGridInfo.minPos = mSpatialGridInfo.minPos;
	mScalarFieldGridInfo.maxPos = mSpatialGridInfo.maxPos;
	mScalarFieldGridInfo.resolution = make_uint3(
		(mScalarFieldGridInfo.maxPos.x - mScalarFieldGridInfo.minPos.x) / mScalarFieldGridInfo.cellSize + 1,
		(mScalarFieldGridInfo.maxPos.y - mScalarFieldGridInfo.minPos.y) / mScalarFieldGridInfo.cellSize + 1,
		(mScalarFieldGridInfo.maxPos.z - mScalarFieldGridInfo.minPos.z) / mScalarFieldGridInfo.cellSize + 1);

	std::cout << "Scalar Field Resolution: " << mScalarFieldGridInfo.resolution.x << " * "
		<< mScalarFieldGridInfo.resolution.y << " * " << mScalarFieldGridInfo.resolution.z << std::endl;

	//! assigment of simulation parameters.
	mSimParam.effR = mSpatialGridInfo.cellSize;
	mSimParam.effRSq = mSimParam.effR * mSimParam.effR;
	mSimParam.scSpGridResRatio = mSpatialGridInfo.cellSize / mScalarFieldGridInfo.cellSize;

	//! copy the particles to gpu.
	mNumParticles = particles.size();
	CUDA_CREATE_GRID_1D(mDeviceParticlesArray, particles.size(), SimpleParticle);
	checkCudaErrors(cudaMemcpy(mDeviceParticlesArray.grid, static_cast<void*>(particles.data()),
		particles.size() * sizeof(SimpleParticle), cudaMemcpyHostToDevice));

	//! spatial hashing grid building.
	spatialHashingGridBuilding(particles);

	//! memory allocation of other arrays.
	CUDA_CREATE_GRID_3D(mDeviceScalarFieldGrid, mScalarFieldGridInfo.resolution, SimpleVertex);
	CUDA_CREATE_GRID_3D(mDeviceIsSurfaceGrid, mScalarFieldGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(mDeviceIsSurfaceGridScan, mScalarFieldGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(mDeviceSurfaceVertexIndexArray, mScalarFieldGridInfo.resolution, uint);
}

void ReconstructorGPU::onEndFrame(unsigned int frameIndex)
{
	//! first of all, save the mesh to file.
	saveFluidSurfaceObjToFile(frameIndex);

	CUDA_DESTROY_GRID(mDeviceParticlesArray);
	CUDA_DESTROY_GRID(mDeviceScalarFieldGrid);
	CUDA_DESTROY_GRID(mDeviceCellParticleIndexArray);

	CUDA_DESTROY_GRID(mDeviceDensityGrid);
	CUDA_DESTROY_GRID(mDeviceIsSurfaceGrid);
	CUDA_DESTROY_GRID(mDeviceIsSurfaceGridScan);
	CUDA_DESTROY_GRID(mDeviceSurfaceVertexIndexArray);

	CUDA_DESTROY_GRID(mDeviceIsValidSurfaceGrid);
	CUDA_DESTROY_GRID(mDeviceIsValidSurfaceGridScan);
	CUDA_DESTROY_GRID(mDeviceValidSurfaceIndexArray);
	CUDA_DESTROY_GRID(mDeviceNumVerticesGrid);
	CUDA_DESTROY_GRID(mDeviceNumVerticesGridScan);

	CUDA_DESTROY_GRID(mDeviceVertexArray);
	CUDA_DESTROY_GRID(mDeviceNormalArray);

}

std::vector<SimpleParticle> ReconstructorGPU::readParticlesFromFile(unsigned int frameIndex)
{
	char basename[256];
	snprintf(basename, sizeof(basename), mFilePattern.c_str(), frameIndex);
	std::string path = mFileDirectory + std::string(basename) + ".xyz";
	std::ifstream file(path.c_str());

	std::vector<SimpleParticle> result;
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
		ss3 >> mSpatialGridInfo.cellSize;

		//! particle radius.
		std::getline(file, line);
		std::stringstream ss4;
		ss4 << line;
		ss4 >> mSimParam.particleRadius;

		//! calculation of resoluation.
		mSpatialGridInfo.resolution.x = (mSpatialGridInfo.maxPos.x - mSpatialGridInfo.minPos.x)
			/ mSpatialGridInfo.cellSize + 1;
		mSpatialGridInfo.resolution.y = (mSpatialGridInfo.maxPos.y - mSpatialGridInfo.minPos.y)
			/ mSpatialGridInfo.cellSize + 1;
		mSpatialGridInfo.resolution.z = (mSpatialGridInfo.maxPos.z - mSpatialGridInfo.minPos.z)
			/ mSpatialGridInfo.cellSize + 1;

		//! particles' positions.
		while (std::getline(file, line))
		{
			std::stringstream str;
			str << line;
			SimpleParticle tmp;
			str >> tmp.pos.x;
			str >> tmp.pos.y;
			str >> tmp.pos.z;
			result.push_back(tmp);
		}

		file.close();

		std::cout << "Finish reading " << path << ".\n";
	}
	else
		std::cout << "Failed to read the file:" << path << std::endl;

	return result;
}

void ReconstructorGPU::spatialHashingGridBuilding(const std::vector<SimpleParticle>& particles)
{
	assert(mSpatialGridInfo.resolution.x > 0 && mSpatialGridInfo.resolution.y > 0 && mSpatialGridInfo.resolution.z > 0);

	//! memory allocation for virtual density grid and spatial hashing grid.
	CUDA_CREATE_GRID_3D(mDeviceDensityGrid, mSpatialGridInfo.resolution, float);
	CUDA_CREATE_GRID_3D(mDeviceCellParticleIndexArray, mSpatialGridInfo.resolution, IndexInfo);
	cudaMemset(mDeviceDensityGrid.grid, 0.0f, mDeviceDensityGrid.size * sizeof(float));
	cudaMemset(mDeviceCellParticleIndexArray.grid, 0xffffffff, mDeviceCellParticleIndexArray.size * sizeof(IndexInfo));

	//! launch the building kernel function.
	launchSpatialGridBuilding(mDeviceParticlesArray, mNumParticles, mDeviceCellParticleIndexArray,
		mDeviceDensityGrid, mSpatialGridInfo);
}

void ReconstructorGPU::saveFluidSurfaceObjToFile(unsigned int frameIndex)
{
	//! get triangles from device.
	std::vector<Triangle> triangles;
	size_t nums = mDeviceVertexArray.size;

	if (nums == 0)
	{
		std::cerr << "Nothing produced.\n";
		return;
	}

	std::vector<float3> positions;
	std::vector<float3> normals;
	positions.resize(nums);
	normals.resize(nums);

	checkCudaErrors(cudaMemcpy(static_cast<void*>(positions.data()), mDeviceVertexArray.grid,
		sizeof(float3) * nums, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(static_cast<void*>(normals.data()), mDeviceNormalArray.grid,
		sizeof(float3) * nums, cudaMemcpyDeviceToHost));

	for (size_t index = 0; index < nums; index += 3)
	{
		Triangle tmp;
		tmp.vertices[0] = fVector3(positions[index + 0].x, positions[index + 0].y, positions[index + 0].z);
		tmp.vertices[1] = fVector3(positions[index + 1].x, positions[index + 1].y, positions[index + 1].z);
		tmp.vertices[2] = fVector3(positions[index + 2].x, positions[index + 2].y, positions[index + 2].z);
		tmp.normals[0] = fVector3(normals[index + 0].x, normals[index + 0].y, normals[index + 0].z);
		tmp.normals[1] = fVector3(normals[index + 1].x, normals[index + 1].y, normals[index + 1].z);
		tmp.normals[2] = fVector3(normals[index + 2].x, normals[index + 2].y, normals[index + 2].z);
		triangles.push_back(tmp);
	}

	std::cout << "Number of vertices : " << nums << std::endl;
	std::cout << "Number of normals  : " << nums << std::endl;
	std::cout << "Number of triangles: " << triangles.size() << std::endl;

	//! save triangles of mesh to .obj file.
	char basename[256];
	snprintf(basename, sizeof(basename), mFilePattern.c_str(), frameIndex);
	std::string path = mFileDirectory + std::string(basename) + ".obj";
	std::ofstream file(path.c_str());

	if (file)
	{
		std::cout << "Writing to " << path << "...\n";

		//! positions.
		for (const auto &elem : triangles)
		{
			file << "v " << elem.vertices[0].x << " " << elem.vertices[0].y << " " << elem.vertices[0].z << std::endl;
			file << "v " << elem.vertices[1].x << " " << elem.vertices[1].y << " " << elem.vertices[1].z << std::endl;
			file << "v " << elem.vertices[2].x << " " << elem.vertices[2].y << " " << elem.vertices[2].z << std::endl;
		}
		//! normals.
		for (const auto &elem : triangles)
		{
			file << "vn " << elem.normals[0].x << " " << elem.normals[0].y << " " << elem.normals[0].z << std::endl;
			file << "vn " << elem.normals[1].x << " " << elem.normals[1].y << " " << elem.normals[1].z << std::endl;
			file << "vn " << elem.normals[2].x << " " << elem.normals[2].y << " " << elem.normals[2].z << std::endl;
		}

		//! faces.
		for (size_t i = 1; i <= triangles.size() * 3; i += 3)
		{
			file << "f ";
			file << (i + 0) << "//" << (i + 0) << " ";
			file << (i + 1) << "//" << (i + 1) << " ";
			file << (i + 2) << "//" << (i + 2) << " ";
			file << std::endl;
		}

		file.close();

		std::cout << "Finish writing " << path << ".\n";
	}
	else
		std::cerr << "Failed to save the file: " << path << std::endl;
	
}

void ReconstructorGPU::saveTimeConsumingRecordToFile()
{
	if (mTimeConsuming.empty())
		return;

	std::string fileName = getAlgorithmType();
	std::string path = mFileDirectory + fileName + ".time";
	std::ofstream file(path.c_str());
	if (file)
	{
		std::cout << "Writing " << path << "...\n";
		for (auto frameIndex = mFrameFrom; frameIndex < mFrameTo; ++frameIndex)
		{
			double record = mTimeConsuming[frameIndex - mFrameFrom];
			file << record << ' ';
		}
		file.close();
	}
	else
		std::cerr << "Failed to save the file: " << path << std::endl;

}

void ReconstructorGPU::onInitialization()
{
	//! timer for recording.
	mTimer = std::shared_ptr<Timer>(new Timer());

	//! isocontour value.
	mSimParam.isoValue = -0.0001f;
	mSimParam.searchExtent = 2;

	//! memory allocation for auxiliary textures.
	cudaMallocMemcpy((void**)&mDeviceEdgeTable, (void*)MarchingCubesHelper::edgeFlags, 256 * sizeof(uint));
	cudaMallocMemcpy((void**)&mDeviceEdgeIndicesOfTriangleTable, (void*)MarchingCubesHelper::edgeIndexesOfTriangle, 256 * 16 * sizeof(int));
	cudaMallocMemcpy((void**)&mDeviceNumVerticesTable, (void*)MarchingCubesHelper::numVertices, 256 * sizeof(uint));
	cudaMallocMemcpy((void**)&mDeviceVertexIndicesOfEdgeTable, (void*)MarchingCubesHelper::vertexIndexesOfEdge, 12 * 2 * sizeof(int));
	bindTextures(mDeviceEdgeTable, mDeviceEdgeIndicesOfTriangleTable, mDeviceNumVerticesTable, mDeviceVertexIndicesOfEdgeTable);

	//! initialization of arrays.
	INITGRID_ZERO(mDeviceParticlesArray);
	INITGRID_ZERO(mDeviceScalarFieldGrid);
	INITGRID_ZERO(mDeviceCellParticleIndexArray);

	INITGRID_ZERO(mDeviceDensityGrid);
	INITGRID_ZERO(mDeviceIsSurfaceGrid);
	INITGRID_ZERO(mDeviceIsSurfaceGridScan);
	INITGRID_ZERO(mDeviceIsValidSurfaceGrid);
	INITGRID_ZERO(mDeviceIsValidSurfaceGridScan);
	INITGRID_ZERO(mDeviceNumVerticesGrid);
	INITGRID_ZERO(mDeviceNumVerticesGridScan);
	INITGRID_ZERO(mDeviceValidSurfaceIndexArray);

	INITGRID_ZERO(mDeviceVertexArray);
	INITGRID_ZERO(mDeviceNormalArray);
}

void ReconstructorGPU::onFinalization()
{
	//! release of auxiliary textures.
	safeCudaFree((void**)&mDeviceEdgeTable);
	safeCudaFree((void**)&mDeviceEdgeIndicesOfTriangleTable);
	safeCudaFree((void**)&mDeviceNumVerticesTable);
	safeCudaFree((void**)&mDeviceVertexIndicesOfEdgeTable);
}
