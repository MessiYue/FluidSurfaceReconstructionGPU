#include "ReconstructorGPUOurs.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CudaUtils.h"
#include "ReconstructionCUDA.cuh"
#include "MarchingCubesHelper.h"

ReconstructorGPUOurs::ReconstructorGPUOurs(
	const std::string & directory,
	const std::string & filePattern,
	unsigned int from, unsigned int to) : 
	ReconstructorGPU(directory, filePattern, from, to) {}

void ReconstructorGPUOurs::saveFluidSurfaceObjToFile(unsigned int frameIndex)
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

void ReconstructorGPUOurs::onBeginFrame(unsigned int frameIndex)
{
	CUDA_CREATE_GRID_3D(mDeviceIsSurfaceGrid, mScalarFieldGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(mDeviceIsSurfaceGridScan, mScalarFieldGridInfo.resolution, uint);
	//CUDA_CREATE_GRID_3D(mDeviceSurfaceVerticesIndexArray, mScalarFieldGridInfo.resolution, uint);
}

void ReconstructorGPUOurs::onFrameMove(unsigned int frameIndex) {}

void ReconstructorGPUOurs::onEndFrame(unsigned int frameIndex)
{
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
}

void ReconstructorGPUOurs::onInitialization()
{
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

void ReconstructorGPUOurs::onFinalization() {}

void ReconstructorGPUOurs::detectionOfValidSurfaceCubes()
{
	if (mNumSurfaceVertices == 0)
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
	launchDetectValidSurfaceCubes(
		gridDim_,
		blockDim_,
		mDeviceSurfaceVerticesIndexArray,
		mNumSurfaceVertices,
		mDeviceScalarFieldGrid,
		mDeviceIsValidSurfaceGrid,
		mDeviceNumVerticesGrid, 
		mDeviceIsSurfaceGrid,
		mSimParam);
}

void ReconstructorGPUOurs::compactationOfValidSurafceCubes()
{
	//! calculation of exclusive prefix sum of mDeviceIsValidSurfaceGrid.
	mNumValidSurfaceCubes = launchThrustExclusivePrefixSumScan(
		mDeviceIsValidSurfaceGridScan.grid,
		mDeviceIsValidSurfaceGrid.grid,
		(uint)mDeviceIsValidSurfaceGrid.size);

	//! calculation of exclusive prefix sum of mDeviceNumVerticesGrid.
	mNumSurfaceMeshVertices = launchThrustExclusivePrefixSumScan(
		mDeviceNumVerticesGridScan.grid,
		mDeviceNumVerticesGrid.grid,
		(uint)mDeviceNumVerticesGrid.size);

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
	launchCompactValidSurfaceCubes(
		gridDim_,
		blockDim_,
		mDeviceValidSurfaceIndexArray,
		mDeviceIsValidSurfaceGridScan,
		mDeviceIsValidSurfaceGrid,
		mSimParam);

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
	launchGenerateTriangles(
		gridDim_, 
		blockDim_,
		mDeviceSurfaceVerticesIndexArray,
		mDeviceValidSurfaceIndexArray,
		mScalarFieldGridInfo,
		mDeviceNumVerticesGridScan,
		mDeviceScalarFieldGrid,
		mDeviceVertexArray,
		mDeviceNormalArray,
		mSimParam);
}
