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

ReconstructorGPU::ReconstructorGPU(const std::string & directory, const std::string & filePattern,
	unsigned int from, unsigned int to) : 
	mFileDirectory(directory), mFilePattern(filePattern),
	mFrameFrom(from), mFrameTo(to), mSaveVisFile(false)
{
}

ReconstructorGPU::~ReconstructorGPU() {}

void ReconstructorGPU::reconstruct()
{
	if (mFrameFrom >= mFrameTo)
	{
		std::cout << "Wrong settings occur for frame range.\n";
		return;
	}

	//! initialization.
	initialization();

	mTimeConsuming.clear();

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

		//! at the end of each frame, do some post-process here.
		endFrame(frameIndex);
	}

	//! save the time consumed to file.
	saveTimeConsumingRecordToFile();

	//! finalization.
	finalization();
}

SimParam ReconstructorGPU::getSimulationParameters() const { return mSimParam; }

unsigned int ReconstructorGPU::getNumberOfParticles() const { return mNumParticles; }

std::vector<double> ReconstructorGPU::getTimeConsumingSequence() const { return mTimeConsuming; }

void ReconstructorGPU::setOutputVisualizeFile(bool flag)
{
	mSaveVisFile = flag;
}

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
	CUDA_CREATE_GRID_3D(mDeviceIsSurfaceGrid, mScalarFieldGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(mDeviceIsSurfaceGridScan, mScalarFieldGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(mDeviceSurfaceVerticesIndexArray, mScalarFieldGridInfo.resolution, uint);

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
	saveFluidSurfaceObjToFile(frameIndex);

	CUDA_DESTROY_GRID(mDeviceParticlesArray);
	CUDA_DESTROY_GRID(mDeviceParticlesDensityArray);
	CUDA_DESTROY_GRID(mDeviceSurfaceParticlesFlagGrid);
	CUDA_DESTROY_GRID(mDeviceScalarFieldGrid);
	CUDA_DESTROY_GRID(mDeviceCellParticleIndexArray);

	CUDA_DESTROY_GRID(mDeviceFlagGrid);
	CUDA_DESTROY_GRID(mDeviceIsSurfaceGrid);
	CUDA_DESTROY_GRID(mDeviceIsSurfaceGridScan);
	CUDA_DESTROY_GRID(mDeviceSurfaceVerticesIndexArray);

	CUDA_DESTROY_GRID(mDeviceIsValidSurfaceGrid);
	CUDA_DESTROY_GRID(mDeviceIsValidSurfaceGridScan);
	CUDA_DESTROY_GRID(mDeviceValidSurfaceIndexArray);
	CUDA_DESTROY_GRID(mDeviceNumVerticesGrid);
	CUDA_DESTROY_GRID(mDeviceNumVerticesGridScan);

	CUDA_DESTROY_GRID(mDeviceVertexArray);
	CUDA_DESTROY_GRID(mDeviceNormalArray);

	//! extra action for sub class.
	onEndFrame(frameIndex);
}

void ReconstructorGPU::readParticlesFromFile(unsigned int frameIndex,
	std::vector<ParticlePosition>& particles, std::vector<ScalarValue>& densities)
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
	assert(mSpatialGridInfo.resolution.x > 0 && mSpatialGridInfo.resolution.y > 0 && mSpatialGridInfo.resolution.z > 0);

	//! memory allocation for virtual density grid and spatial hashing grid.
	CUDA_CREATE_GRID_3D(mDeviceFlagGrid, mSpatialGridInfo.resolution, float);
	CUDA_CREATE_GRID_3D(mDeviceCellParticleIndexArray, mSpatialGridInfo.resolution, IndexRange);
	cudaMemset(mDeviceFlagGrid.grid, 0.0f, mDeviceFlagGrid.size * sizeof(float));
	cudaMemset(mDeviceCellParticleIndexArray.grid, 0xffffffff, mDeviceCellParticleIndexArray.size * sizeof(IndexRange));

	//! launch the building kernel function.
	launchSpatialGridBuilding(mDeviceParticlesArray, mDeviceParticlesDensityArray, mNumParticles, 
		mDeviceCellParticleIndexArray, mDeviceFlagGrid, mSpatialGridInfo);
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

void ReconstructorGPU::initialization()
{
	//! timer for recording.
	mTimer = std::shared_ptr<Timer>(new Timer());

	mSimParam.spatialCellSizeScale = 1.0;

	//! memory allocation for auxiliary textures.
	cudaMallocMemcpy((void**)&mDeviceEdgeTable, (void*)MarchingCubesHelper::edgeFlags, 256 * sizeof(uint));
	cudaMallocMemcpy((void**)&mDeviceEdgeIndicesOfTriangleTable, (void*)MarchingCubesHelper::edgeIndexesOfTriangle, 256 * 16 * sizeof(int));
	cudaMallocMemcpy((void**)&mDeviceNumVerticesTable, (void*)MarchingCubesHelper::numVertices, 256 * sizeof(uint));
	cudaMallocMemcpy((void**)&mDeviceVertexIndicesOfEdgeTable, (void*)MarchingCubesHelper::vertexIndexesOfEdge, 12 * 2 * sizeof(int));
	bindTextures(mDeviceEdgeTable, mDeviceEdgeIndicesOfTriangleTable, mDeviceNumVerticesTable, mDeviceVertexIndicesOfEdgeTable);

	//! initialization of arrays.
	INITGRID_ZERO(mDeviceParticlesArray);
	INITGRID_ZERO(mDeviceParticlesDensityArray);
	INITGRID_ZERO(mDeviceScalarFieldGrid);
	INITGRID_ZERO(mDeviceCellParticleIndexArray);

	INITGRID_ZERO(mDeviceFlagGrid);
	INITGRID_ZERO(mDeviceIsSurfaceGrid);
	INITGRID_ZERO(mDeviceIsSurfaceGridScan);
	INITGRID_ZERO(mDeviceIsValidSurfaceGrid);
	INITGRID_ZERO(mDeviceIsValidSurfaceGridScan);
	INITGRID_ZERO(mDeviceNumVerticesGrid);
	INITGRID_ZERO(mDeviceNumVerticesGridScan);
	INITGRID_ZERO(mDeviceValidSurfaceIndexArray);

	INITGRID_ZERO(mDeviceVertexArray);
	INITGRID_ZERO(mDeviceNormalArray);

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

void ReconstructorGPU::detectionOfValidSurfaceCubes()
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
	launchDetectValidSurfaceCubes(gridDim_, blockDim_, mDeviceSurfaceVerticesIndexArray, mNumSurfaceVertices,
		mDeviceScalarFieldGrid, mDeviceIsValidSurfaceGrid, mDeviceNumVerticesGrid, mDeviceIsSurfaceGrid, mSimParam);
}

void ReconstructorGPU::compactationOfValidSurafceCubes()
{
	//! calculation of exclusive prefix sum of mDeviceIsValidSurfaceGrid.
	mNumValidSurfaceCubes = launchThrustExclusivePrefixSumScan(mDeviceIsValidSurfaceGridScan.grid,
		mDeviceIsValidSurfaceGrid.grid, (uint)mDeviceIsValidSurfaceGrid.size);

	//! calculation of exclusive prefix sum of mDeviceNumVerticesGrid.
	mNumSurfaceMeshVertices = launchThrustExclusivePrefixSumScan(mDeviceNumVerticesGridScan.grid,
		mDeviceNumVerticesGrid.grid, (uint)mDeviceNumVerticesGrid.size);

	if (mNumSurfaceMeshVertices <= 0)
	{
		std::cerr << "No vertex of surface mesh detected!\n";
		return;
	}

	std::cout << "valid surface cubes ratio: " << static_cast<double>(mNumValidSurfaceCubes)
		/ (mScalarFieldGridInfo.resolution.x * mScalarFieldGridInfo.resolution.y * mScalarFieldGridInfo.resolution.z) << std::endl;

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

void ReconstructorGPU::generationOfSurfaceMeshUsingMC()
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
	launchGenerateTriangles(gridDim_, blockDim_, mDeviceSurfaceVerticesIndexArray, mDeviceValidSurfaceIndexArray,
		mScalarFieldGridInfo, mDeviceNumVerticesGridScan, mDeviceScalarFieldGrid, mDeviceVertexArray,
		mDeviceNormalArray, mSimParam);
}
