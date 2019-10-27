#include "ReconstructorGPUAkinci12.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CudaUtils.h"
#include "MarchingCubesCPU.h"
#include "ReconstructionCUDA.cuh"

ReconstructorGPUAkinci12::ReconstructorGPUAkinci12(
	const std::string & directory,
	const std::string & filePattern,
	unsigned int from, unsigned int to) : 
	ReconstructorGPU(directory, filePattern, from, to) {}

std::string ReconstructorGPUAkinci12::getAlgorithmType()
{
	return std::string("Akinci12 using ZB05 kernel");
}

void ReconstructorGPUAkinci12::onBeginFrame(unsigned int frameIndex)
{
	CUDA_CREATE_GRID_3D(mDeviceIsSurfaceGrid, mScalarFieldGridInfo.resolution, uint);
	mVertexArray.clear();
	mNormalArray.clear();
	mHostSurfaceVerticesIndexArray.clear();
	mValidSurfaceCubesIndexArray.clear();
}

void ReconstructorGPUAkinci12::onFrameMove(unsigned int frameIndex)
{
	//! step1: extraction of surface particles.
	std::cout << "step1, extraction of surface particles...\n";
	extractionOfSurfaceParticles();

	//! step2: extraction of surface vertices.
	std::cout << "step2, extraction of surface vertices...\n";
	extractionOfSurfaceVertices();

	//! step3: compactation of surface vertices.
	std::cout << "step3, compactation of surface vertices...\n";
	compactationOfSurfaceVertices();

	//! step4: computation of scalar field grid.
	std::cout << "step4, computation of scalar field grid...\n";
	computationOfScalarFieldGrid();

	//! step5: detection of valid surface cubes.
	std::cout << "step5, detection of valid surface cubes...\n";
	detectionOfValidSurfaceCubes();

	//! step6: generation of triangles for surface mesh.
	std::cout << "step6, generation of triangles for surface mesh...\n";
	generationOfSurfaceMeshUsingMC();

}

void ReconstructorGPUAkinci12::onEndFrame(unsigned int frameIndex)
{
	CUDA_DESTROY_GRID(mDeviceIsSurfaceGrid);
	CUDA_DESTROY_GRID(mDeviceIsValidSurfaceGrid);
	CUDA_DESTROY_GRID(mDeviceSurfaceVerticesIndexArray);
}

void ReconstructorGPUAkinci12::onInitialization()
{
	INITGRID_ZERO(mDeviceIsSurfaceGrid);
	INITGRID_ZERO(mDeviceIsValidSurfaceGrid);
	INITGRID_ZERO(mDeviceSurfaceVerticesIndexArray);

	//! isocontour value.
	mSimParam.isoValue = -0.0001f;
	//! search extent.
	mSimParam.expandExtent = 4;
	mSimParam.scSpGridResRatio = 4;
	mSimParam.spatialCellSizeScale = 1.0;

}

void ReconstructorGPUAkinci12::onFinalization() {}

void ReconstructorGPUAkinci12::extractionOfSurfaceParticles()
{
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumParticles, gridDim_, blockDim_);

	//! initialization for surface particle flag.
	checkCudaErrors(cudaMemset(mDeviceSurfaceParticlesFlagGrid.grid, 0,
		mNumParticles * sizeof(uint)));

	//! launch the extraction function.
	launchExtractionOfSurfaceParticlesForAkinci(
		gridDim_,
		blockDim_,
		mSimParam,
		mDeviceParticlesArray,
		mDeviceParticlesDensityArray,
		mDeviceCellParticleIndexArray,
		mSpatialGridInfo,
		mDeviceSurfaceParticlesFlagGrid);
	getLastCudaError("launch extractionOfSurfaceParticles() failed");
}

void ReconstructorGPUAkinci12::extractionOfSurfaceVertices()
{
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
	getLastCudaError("launch extractionOfSurfaceVertices() failed");

}

void ReconstructorGPUAkinci12::compactationOfSurfaceVertices()
{
	//! the cpu serial part of alogirthm proposed by Akinci etc.
	mNumSurfaceVertices = 0;

	std::vector<uint> isSurfaceArray;
	isSurfaceArray.resize(mDeviceIsSurfaceGrid.size, 0);
	checkCudaErrors(cudaMemcpy(static_cast<void*>(isSurfaceArray.data()),
		mDeviceIsSurfaceGrid.grid, sizeof(uint) * mDeviceIsSurfaceGrid.size, cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < isSurfaceArray.size(); ++i)
	{
		if (isSurfaceArray[i] == 1)
		{
			mHostSurfaceVerticesIndexArray.push_back(i);
		}
	}
	mNumSurfaceVertices = mHostSurfaceVerticesIndexArray.size();

	if (mNumSurfaceVertices == 0)
		std::cerr << "No surface vertex detected!\n";
	
	//! copy to gpu.
	CUDA_CREATE_GRID_1D_SET(mDeviceSurfaceVerticesIndexArray, mNumSurfaceVertices,
		mNumSurfaceVertices, 0, uint);
	checkCudaErrors(cudaMemcpy(mDeviceSurfaceVerticesIndexArray.grid,
		mHostSurfaceVerticesIndexArray.data(), sizeof(uint) * mNumSurfaceVertices, cudaMemcpyHostToDevice));

}

void ReconstructorGPUAkinci12::computationOfScalarFieldGrid()
{
	if (mNumSurfaceVertices == 0)
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

void ReconstructorGPUAkinci12::detectionOfValidSurfaceCubes()
{
	if (mNumSurfaceVertices == 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}

	//! memory allocation for detection of valid surface cubes.
	CUDA_CREATE_GRID_1D_SET(mDeviceIsValidSurfaceGrid, mNumSurfaceVertices, mNumSurfaceVertices, 0, uint);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumSurfaceVertices, gridDim_, blockDim_);

	//! launch the detection kernel function.
	launchDetectionOfValidSurfaceCubesForAkinci(
		gridDim_,
		blockDim_,
		mDeviceSurfaceVerticesIndexArray,
		mNumSurfaceVertices,
		mDeviceScalarFieldGrid,
		mDeviceIsValidSurfaceGrid,
		mDeviceIsSurfaceGrid,
		mSimParam);
	getLastCudaError("launch detectionOfValidSurfaceCubes() failed");
}

void ReconstructorGPUAkinci12::generationOfSurfaceMeshUsingMC()
{
	mNumValidSurfaceCubes = 0;

	//! copy valid cubes' indices from gpu.
	std::vector<uint> validVerticesIndexArray;
	std::vector<uint> isValidSurfaceArray;
	isValidSurfaceArray.resize(mNumSurfaceVertices, 0);
	checkCudaErrors(cudaMemcpy(static_cast<void*>(isValidSurfaceArray.data()),
		mDeviceIsValidSurfaceGrid.grid, sizeof(uint) * mNumSurfaceVertices, cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < isValidSurfaceArray.size(); ++i)
	{
		if (isValidSurfaceArray[i] == 1)
		{
			validVerticesIndexArray.push_back(mHostSurfaceVerticesIndexArray[i]);
			mValidSurfaceCubesIndexArray.push_back(i);
		}
	}
	mNumValidSurfaceCubes = validVerticesIndexArray.size();

	//! copy scalar field grid from gpu.
	std::vector<ScalarValue> scalarGrid;
	scalarGrid.resize(mDeviceScalarFieldGrid.size);
	checkCudaErrors(cudaMemcpy(static_cast<void*>(scalarGrid.data()),
		mDeviceScalarFieldGrid.grid, sizeof(ScalarValue) * scalarGrid.size(), cudaMemcpyDeviceToHost));

	//! perform marching cubes here.
	MarchingCubesCPU mc(&scalarGrid, mScalarFieldGridInfo, mSimParam.isoValue);
	for (size_t i = 0; i < validVerticesIndexArray.size(); ++i)
	{
		int index1D = validVerticesIndexArray[i];
		iVector3 index3D = mc.index1DTo3D(index1D);
		Triangle triangles[5];
		int triCount = 0;
		//! marching cube algorithm.
		mc.marchingCubes(index3D, triangles, triCount);
		for (size_t i = 0; i < triCount; ++i)
		{
			mVertexArray.push_back(triangles[i].vertices[0]);
			mVertexArray.push_back(triangles[i].vertices[1]);
			mVertexArray.push_back(triangles[i].vertices[2]);

			mNormalArray.push_back(triangles[i].normals[0]);
			mNormalArray.push_back(triangles[i].normals[1]);
			mNormalArray.push_back(triangles[i].normals[2]);
		}
	}
}

void ReconstructorGPUAkinci12::saveMiddleDataToVisFile(unsigned int frameIndex)
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

		//! particle radius.
		file << mSimParam.particleRadius << std::endl;

		//! particles.
		file << mNumParticles << std::endl;
		std::vector<ParticlePosition> rawParticles;
		rawParticles.resize(mNumParticles);
		checkCudaErrors(cudaMemcpy(static_cast<void*>(rawParticles.data()),
			mDeviceParticlesArray.grid, sizeof(ParticlePosition) * mNumParticles, cudaMemcpyDeviceToHost));
		for (size_t i = 0; i < mNumParticles; ++i)
		{
			file << rawParticles[i].pos.x << ' ' << rawParticles[i].pos.y << ' '
				<< rawParticles[i].pos.z << std::endl;
		}
		std::vector<ParticlePosition>().swap(rawParticles);

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
		if (mNumSurfaceVertices > 0)
			file << std::endl;
		std::vector<uint>().swap(surfaceVerticesIndexArray);

		//! surface particles.
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
		for (size_t i = 0; i < mNumValidSurfaceCubes; ++i)
			file << mValidSurfaceCubesIndexArray[i] << ' ';
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

void ReconstructorGPUAkinci12::saveFluidSurfaceObjToFile(unsigned int frameIndex)
{
	//! get triangles from device.
	std::vector<Triangle> triangles;
	size_t nums = mVertexArray.size();

	for (size_t index = 0; index < nums; index += 3)
	{
		Triangle tmp;
		tmp.vertices[0] = fVector3(mVertexArray[index + 0].x, mVertexArray[index + 0].y, mVertexArray[index + 0].z);
		tmp.vertices[1] = fVector3(mVertexArray[index + 1].x, mVertexArray[index + 1].y, mVertexArray[index + 1].z);
		tmp.vertices[2] = fVector3(mVertexArray[index + 2].x, mVertexArray[index + 2].y, mVertexArray[index + 2].z);
		tmp.normals[0] = fVector3(mNormalArray[index + 0].x, mNormalArray[index + 0].y, mNormalArray[index + 0].z);
		tmp.normals[1] = fVector3(mNormalArray[index + 1].x, mNormalArray[index + 1].y, mNormalArray[index + 1].z);
		tmp.normals[2] = fVector3(mNormalArray[index + 2].x, mNormalArray[index + 2].y, mNormalArray[index + 2].z);
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
