#include "SurfaceReconstructorCUDA.h"

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_runtime_api.h>

#include "Utils.h"
#include "Kernel.cuh"
#include "SurfaceReconstructionCUDA.cuh"
#include "../CPU/MarchingCubesHelper.h"

SurfaceReconstructorCUDA::SurfaceReconstructorCUDA()
{
	// default initialization.

	INITGRID_ZERO(d_DensityGrid);
	INITGRID_ZERO(d_ParticleIndexInfoGrid);
	INITGRID_ZERO(d_ParticleArray);
	INITGRID_ZERO(d_VertexGrid);
	INITGRID_ZERO(d_IsSurfaceGrid);

	INITGRID_ZERO(d_IsValidSurfaceGrid);
	INITGRID_ZERO(d_IsValidSurfaceGridScan);
	INITGRID_ZERO(d_numVerticesGrid);
	INITGRID_ZERO(d_numVerticesGridScan);

	INITGRID_ZERO(d_validSurfaceIndexArray);
	INITGRID_ZERO(d_posGrid);
	INITGRID_ZERO(d_norGrid);

	d_edgeTable = 0;
	d_numVerticesTable = 0;
	d_edgeIndicesOfTriangleTable = 0;
	d_vertexIndicesOfEdgeTable = 0;

}

SurfaceReconstructorCUDA::~SurfaceReconstructorCUDA()
{
	onDestory();
}

void SurfaceReconstructorCUDA::onInitialize(const std::string &path)
{
	//! read particles from given file.
	std::vector<SimpleParticle> particles;
	particles = Utils::readParticlesFromXYZ(path, &mSpGridInfo, mSimParam.particleRadius);
	mScGridInfo.cellSize = mSpGridInfo.cellSize * 1.0;
	mScGridInfo.minPos = mSpGridInfo.minPos;
	mScGridInfo.maxPos = mSpGridInfo.maxPos;
	mScGridInfo.resolution = make_uint3(
		(mScGridInfo.maxPos.x - mScGridInfo.minPos.x) / mScGridInfo.cellSize + 1,
		(mScGridInfo.maxPos.y - mScGridInfo.minPos.y) / mScGridInfo.cellSize + 1,
		(mScGridInfo.maxPos.z - mScGridInfo.minPos.z) / mScGridInfo.cellSize + 1);

	//! initialization of simulation parameters.
	mSimParam.effR = mSpGridInfo.cellSize;
	mSimParam.effRSq = mSimParam.effR * mSimParam.effR;
	mSimParam.isoValue = 0.5f;
	mSimParam.scSpGridResRatio = mSpGridInfo.cellSize / mScGridInfo.cellSize;

	//! copy to gpu.
	mNumParticles = particles.size();
	CUDA_DESTROY_GRID(d_ParticleArray);
	CUDA_CREATE_GRID_1D(d_ParticleArray, particles.size(), SimpleParticle);
	checkCudaErrors(cudaMemcpy(d_ParticleArray.grid, static_cast<void*>(particles.data()),
		particles.size() * sizeof(SimpleParticle), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	//! creation of spatial hashing grid.
	CUDA_CREATE_GRID_3D(d_DensityGrid, mSpGridInfo.resolution, float);
	CUDA_CREATE_GRID_3D(d_ParticleIndexInfoGrid, mSpGridInfo.resolution, IndexInfo);
	launchSpatialGridBuilding(&d_ParticleArray, mNumParticles,
		&d_ParticleIndexInfoGrid, &d_DensityGrid, mSpGridInfo);

	//! creation of auxiliary textures.
	createTextures();

	//! memory allocation of other arrays.
	CUDA_CREATE_GRID_3D(d_IsSurfaceGrid, mScGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(d_IsSurfaceGridScan, mScGridInfo.resolution, uint);
	CUDA_CREATE_GRID_3D(d_VertexGrid, mScGridInfo.resolution, SimpleVertex);
	CUDA_CREATE_GRID_3D(d_SurfaceVertexIndexArray, mScGridInfo.resolution, uint);

	initSimParam(&mSimParam);
}

void SurfaceReconstructorCUDA::onFrameMove()
{
	std::cout << "Begin to construct the fluid surface..\n";
	//	reAllocMemIfNecessary2016();

	//! step1: estimation of surface vertices.
	estimateSurfaceVertex();

	//! step2: compact surface vertices into continuous array.
	compactSurfaceVertex();

	//! step3: calculation of scalar field grid with compacted surface vertices.
	updateScalarGridValuesCompacted();

	//! step4: detection of valid surface cubes.
	detectValidSurfaceCubes();

	//! step5: compactation of valid surface cubes.
	compactValidSurafceCubes();

	//! step6: generation of triangles for surface.
	generateTriangles();

	std::cout << "Fluid surface reconstruction over...\n";
}

void SurfaceReconstructorCUDA::onDestory()
{
	mNumSurfaceVertex = 0;

	safeCudaFree((void**)&d_edgeTable);
	safeCudaFree((void**)&d_edgeIndicesOfTriangleTable);
	safeCudaFree((void**)&d_numVerticesTable);
	safeCudaFree((void**)&d_vertexIndicesOfEdgeTable);

	CUDA_DESTROY_GRID(d_DensityGrid);
	CUDA_DESTROY_GRID(d_ParticleIndexInfoGrid);

	CUDA_DESTROY_GRID(d_IsSurfaceGrid);
	CUDA_DESTROY_GRID(d_ParticleArray);
	CUDA_DESTROY_GRID(d_VertexGrid);
	CUDA_DESTROY_GRID(d_IsSurfaceGridScan);
	CUDA_DESTROY_GRID(d_SurfaceVertexIndexArray);

	CUDA_DESTROY_GRID(d_IsValidSurfaceGrid);
	CUDA_DESTROY_GRID(d_IsValidSurfaceGridScan);
	CUDA_DESTROY_GRID(d_numVerticesGrid);
	CUDA_DESTROY_GRID(d_numVerticesGridScan);
	CUDA_DESTROY_GRID(d_validSurfaceIndexArray);
}

void SurfaceReconstructorCUDA::estimateSurfaceVertex()
{
	//! calculation of virtual density field.
	//mSpatialGrid->updateDensityArray();

	std::cout << "step1: estimation of surface vertices....\n";

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	if (!calcGridDimBlockDim(d_DensityGrid.size, gridDim_, blockDim_))
		return;

	//! set zero for d_IsSurfaceGrid grid.
	checkCudaErrors(cudaMemset(d_IsSurfaceGrid.grid, 0, d_IsSurfaceGrid.size * sizeof(uint)));

	int scExt = 2;
	launchEstimateSurfaceVertices(gridDim_, blockDim_, d_DensityGrid, d_IsSurfaceGrid, scExt, mSimParam);
	getLastCudaError("launchEstimateSurfaceVertices() failed");

}

void SurfaceReconstructorCUDA::updateScalarGridValues()
{
	//mSpatialGrid->updateParticleArrayParticleIndexInfoArray();

	std::cout << "updateScalarGridValues...\n";

	dim3 gridDim_, blockDim_;
	if (!calcGridDimBlockDim(d_VertexGrid.size, gridDim_, blockDim_))
		return;

	//checkCudaErrors(cudaMemcpy(d_ParticleIndexInfoGrid.grid, mSpatialGrid->getParticleIndexInfoArray(),
	//	d_ParticleIndexInfoGrid.size * sizeof(IndexInfo), cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMemcpy(d_ParticleArray.grid, mSpatialGrid->getParticleArray(),
	//	d_ParticleArray.size * sizeof(SimpleParticle), cudaMemcpyHostToDevice));

	//! set zero for scalar field grid.
	checkCudaErrors(cudaMemset(d_VertexGrid.grid, 0, d_VertexGrid.size * sizeof(SimpleVertex)));

	launchUpdateScalarGridValues(gridDim_, blockDim_, d_IsSurfaceGrid, d_ParticleIndexInfoGrid,
		d_ParticleArray, d_VertexGrid, mSpGridInfo, mScGridInfo, mSimParam);
	getLastCudaError("launchUpdateScalarValues failed");

}

void SurfaceReconstructorCUDA::compactSurfaceVertex()
{
	std::cout << "step2: compact surface vertices into continuous array...\n";

	//! calculation of exculsive prefix sum of d_IsSurfaceGrid.
	mNumSurfaceVertex = ThrustExclusiveScanWrapper(d_IsSurfaceGridScan.grid, d_IsSurfaceGrid.grid,
		(uint)d_IsSurfaceGrid.size);
	std::cout << " mNumSurfaceVertex -> " << mNumSurfaceVertex << std::endl;

	if (mNumSurfaceVertex <= 0)
		return;

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(d_IsSurfaceGrid.size, gridDim_, blockDim_);

	//! compactation of surface vertices.
	launchCompactSurfaceVertex(gridDim_, blockDim_, d_SurfaceVertexIndexArray, d_IsSurfaceGridScan,
		d_IsSurfaceGrid, mSimParam);
	getLastCudaError("launchCompactSurfaceVertex failed");

}

void SurfaceReconstructorCUDA::updateScalarGridValuesCompacted()
{
	std::cout << "step3: calculation of scalar field grid with compacted surface vertices...\n";

	if (mNumSurfaceVertex <= 0)
		return;
	
	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumSurfaceVertex, gridDim_, blockDim_);

	//! set zero for scalar field grid.
	checkCudaErrors(cudaMemset(d_VertexGrid.grid, 0, d_VertexGrid.size * sizeof(SimpleVertex)));

	//! calculation of scalar field with compacted surface vertices.
	launchUpdateScalarGridValuesCompacted(gridDim_, blockDim_, d_SurfaceVertexIndexArray, mNumSurfaceVertex,
		d_ParticleIndexInfoGrid, d_ParticleArray, d_VertexGrid, mSpGridInfo, mScGridInfo, mSimParam);
	getLastCudaError("launchUpdateScalarGridValuesCompacted failed");

}

void SurfaceReconstructorCUDA::detectValidSurfaceCubes()
{
	if (mNumSurfaceVertex <= 0)
		return;

	std::cout << "step4: detection of valid surface cubes...\n";

	//! memory allocation for detection of valid surface cubes.
	CUDA_DESTROY_GRID(d_numVerticesGrid);
	CUDA_DESTROY_GRID(d_IsValidSurfaceGrid);
	CUDA_DESTROY_GRID(d_numVerticesGridScan);
	CUDA_DESTROY_GRID(d_IsValidSurfaceGridScan);
	CUDA_CREATE_GRID_1D_SET(d_IsValidSurfaceGrid, mNumSurfaceVertex, mNumSurfaceVertex, 0, uint);
	CUDA_CREATE_GRID_1D_SET(d_IsValidSurfaceGridScan, mNumSurfaceVertex, mNumSurfaceVertex, 0, uint);
	CUDA_CREATE_GRID_1D_SET(d_numVerticesGrid, mNumSurfaceVertex, mNumSurfaceVertex, 0, uint);
	CUDA_CREATE_GRID_1D_SET(d_numVerticesGridScan, mNumSurfaceVertex, mNumSurfaceVertex, 0, uint);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumSurfaceVertex, gridDim_, blockDim_);

	//! detection of valid surface cubes.
	launchDetectValidSurfaceCubes(gridDim_, blockDim_, d_SurfaceVertexIndexArray, mNumSurfaceVertex,
		d_VertexGrid, d_IsValidSurfaceGrid, d_numVerticesGrid, d_IsSurfaceGrid, mSimParam);

}

void SurfaceReconstructorCUDA::compactValidSurafceCubes()
{
	std::cout << "step5: compactation of valid surface cubes...\n";

	//! compactation of valid surface cubes.
	
	//! calculation of exculsive prefix sum of d_IsValidSurfaceGrid.
	mNumValidSurfaceCubes = ThrustExclusiveScanWrapper(d_IsValidSurfaceGridScan.grid,
		d_IsValidSurfaceGrid.grid, (uint)d_IsValidSurfaceGrid.size);

	std::cout << "mNumValidSurfaceCubes->" << mNumValidSurfaceCubes << std::endl;

	//! calculation of exculsive prefix sum of d_numVerticesGrid.
	mNumValidVertices = ThrustExclusiveScanWrapper(d_numVerticesGridScan.grid,
		d_numVerticesGrid.grid, (uint)d_numVerticesGrid.size);

	std::cout << "mNumValidVertices->" << mNumValidVertices << std::endl;

	if (mNumValidVertices <= 0)
		return;

	//£¡memory allocation of valid surface cubes.
	SAFE_CUDA_FREE_GRID(d_validSurfaceIndexArray);
	CUDA_CREATE_GRID_1D_SET(d_validSurfaceIndexArray, mNumValidSurfaceCubes, mNumValidSurfaceCubes, 0, uint);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumSurfaceVertex, gridDim_, blockDim_);

	//! compactation of valid surface cubes.
	launchCompactValidSurfaceCubes(gridDim_, blockDim_, d_validSurfaceIndexArray,
		d_IsValidSurfaceGridScan, d_IsValidSurfaceGrid, mSimParam);

}

void SurfaceReconstructorCUDA::generateTriangles()
{
	if (mNumValidVertices <= 0)
		return;

	std::cout << "step6: generation of triangles for surface...\n";

	//! memory allocation for generation of triangles for surface.
	CUDA_DESTROY_GRID(d_posGrid);
	CUDA_DESTROY_GRID(d_norGrid);
	CUDA_CREATE_GRID_1D_SET(d_posGrid, mNumValidVertices, mNumValidVertices, 0, float3);
	CUDA_CREATE_GRID_1D_SET(d_norGrid, mNumValidVertices, mNumValidVertices, 0, float3);

	//! calculation of grid dim and block dim.
	dim3 gridDim_, blockDim_;
	calcGridDimBlockDim(mNumValidSurfaceCubes, gridDim_, blockDim_);

	//! generation of triangles for surface.
	launchGenerateTriangles(gridDim_, blockDim_, d_SurfaceVertexIndexArray, d_validSurfaceIndexArray,
		mScGridInfo, d_numVerticesGridScan, d_VertexGrid, d_posGrid, d_norGrid, mSimParam);

}

std::vector<Triangle> SurfaceReconstructorCUDA::getTriangles()
{
	//! get triangles from device.

	std::vector<Triangle> triangles;
	size_t nums = d_posGrid.size;
	std::vector<float3> positions;
	std::vector<float3> normals;
	positions.resize(nums);
	normals.resize(nums);

	checkCudaErrors(cudaMemcpy(static_cast<void*>(positions.data()), d_posGrid.grid, sizeof(float3) * nums,
		cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(static_cast<void*>(normals.data()), d_norGrid.grid, sizeof(float3) * nums,
		cudaMemcpyDeviceToHost));

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

	return triangles;
}

void SurfaceReconstructorCUDA::createTextures()
{
	//! memory allocation for auxiliary textures.

	cudaMallocMemcpy(
		(void**)&d_edgeTable,
		(void*)MarchingCubesHelper::edgeFlags,
		256 * sizeof(uint));

	cudaMallocMemcpy(
		(void**)&d_edgeIndicesOfTriangleTable,
		(void*)MarchingCubesHelper::edgeIndexesOfTriangle,
		256 * 16 * sizeof(int));

	cudaMallocMemcpy(
		(void**)&d_numVerticesTable,
		(void*)MarchingCubesHelper::numVertices,
		256 * sizeof(uint));

	cudaMallocMemcpy(
		(void**)&d_vertexIndicesOfEdgeTable,
		(void*)MarchingCubesHelper::vertexIndexesOfEdge,
		12 * 2 * sizeof(int));

	//! finally textures binding.

	bindTextures(d_edgeTable, d_edgeIndicesOfTriangleTable, d_numVerticesTable, d_vertexIndicesOfEdgeTable);
}
