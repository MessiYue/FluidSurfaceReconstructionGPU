
#include <Windows.h>
#include "SurfaceReconstructionCUDA.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <cooperative_groups.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

#include "Defines.h"
#include "Kernel.cuh"
#include "CudaUtils.h"
#include "../CPU/MarchingCubesHelper.h"

using namespace cooperative_groups;

__constant__ float EPSILON_ = (float)1.0e-7;

//! [标记8个顶点与等值面位置关系的8位二进制数]，值为标记12条边与等值面相交与否的整数（只用低12位），大小为256
texture <uint, 1, cudaReadModeElementType> edgeTex;
//! [标记8个顶点与等值面位置关系的8位二进制数][最多15个顶点（三个三角形），
//! 最后一个元素用于结束标记]，值为产生的三角形顶点所在cube的边号
//! 三角形三个顶点的连接顺序为逆时针方向， 大小为256
texture <int, 1, cudaReadModeElementType> edgeIndexesOfTriangleTex;
//! 对应cell/voxel产生的顶点数，大小为256
texture <uint, 1, cudaReadModeElementType> numVerticesTex;
//! [边编号][两个顶点]，值为该边的两个顶点编号，大小为12
texture <uint, 1, cudaReadModeElementType> vertexIndexesOfEdgeTex;

//! func: get global thread id.
inline __device__  
uint getThreadIdGlobal()
{
	uint blockId = blockIdx.z*gridDim.y*gridDim.x + blockIdx.y*gridDim.x + blockIdx.x;
	uint threadId = threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x 
		+ blockId*blockDim.x*blockDim.y*blockDim.z;
	return threadId;
}

//! func: bind auxiliary textures for marching cubes.
extern "C"
void bindTextures(uint* d_edgeTable, int* d_edgeIndicesOfTriangleTable, 
	uint* d_numVerticesTable, uint* d_vertexIndicesOfEdgeTable)
{
	// texture's channel format.
	cudaChannelFormatDesc channelDescUnsigned = 
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc channelDescSigned =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);

	// data transfer and textures binding.
	checkCudaErrors(cudaBindTexture(0, edgeTex, d_edgeTable, channelDescUnsigned));
	checkCudaErrors(cudaBindTexture(0, edgeIndexesOfTriangleTex, d_edgeIndicesOfTriangleTable, channelDescSigned));
	checkCudaErrors(cudaBindTexture(0, numVerticesTex, d_numVerticesTable, channelDescUnsigned));
	checkCudaErrors(cudaBindTexture(0, vertexIndexesOfEdgeTex, d_vertexIndicesOfEdgeTable, channelDescUnsigned));
}

//! --------------------------------------cuda kernel functions------------------------------------------------

//! func: surface vertices' estimation using simple virutal density field.
__global__ 
void estimateSurfaceVertices(
	DensityGrid densityGrid,		// input, virtual density field.
	IsSurfaceGrid isSurfaceGrid,	// output, whether the corresponding grid point is in surface region or not. 
	uint scSearchExt,				// input, search expansion extent.
	SimParam params)				
{
	// get corresponding 3D index.
	uint threadId = getThreadIdGlobal();
	uint3 densityGridRes = densityGrid.resolution;
	uint3 dR_1 = densityGridRes - 1;
	uint3 curIndex3D = index1DTo3D(threadId, densityGridRes);

	// boundary detection.
	if (curIndex3D.x >= dR_1.x || curIndex3D.y >= dR_1.y || curIndex3D.z >= dR_1.z)
		return;

	// get corresponding situation flag.
	uint vertexFlag = getVertexFlag(curIndex3D, densityGrid, 0.5f);

	// 当前cube与等值面有交点
	if (vertexFlag > 0 && vertexFlag < 255)
	{
		int3 minIndex3D = make_int3(curIndex3D.x, curIndex3D.y, curIndex3D.z);
		int3 maxIndex3D = make_int3(curIndex3D.x + 1, curIndex3D.y + 1, curIndex3D.z + 1);

		// expanding.
		minIndex3D = minIndex3D * params.scSpGridResRatio - scSearchExt;
		maxIndex3D = maxIndex3D * params.scSpGridResRatio + scSearchExt;
		
		// clamping.
		minIndex3D = clamp(minIndex3D, make_int3(0, 0, 0), 
			make_int3(isSurfaceGrid.resolution.x - 1, isSurfaceGrid.resolution.y - 1, isSurfaceGrid.resolution.z - 1));
		maxIndex3D = clamp(maxIndex3D, make_int3(0, 0, 0), 
			make_int3(isSurfaceGrid.resolution.x - 1, isSurfaceGrid.resolution.y - 1, isSurfaceGrid.resolution.z - 1));

		// mark corresponding cell as surface cell (let it equals 1).
		for (uint zSc = minIndex3D.z; zSc <= maxIndex3D.z; zSc++)
		{
			for (uint ySc = minIndex3D.y; ySc <= maxIndex3D.y; ySc++)
			{
				for (uint xSc = minIndex3D.x; xSc <= maxIndex3D.x; xSc++)
				{
					uint3 curIndex3DInScalarGrid = make_uint3(xSc, ySc, zSc);
					isSurfaceGrid.grid[index3DTo1D(curIndex3DInScalarGrid, isSurfaceGrid.resolution)] = 1;
				}
			}
		}
	}
}

//! func: calculate the corresponding vertex's scalar value of scalar field using TMC01 method.
__device__ 
void updateScalarFieldValueTC01(
	uint vertexIndex1D,								// input, vertex's index of scalar field grid.
	ParticleIndexInfoGrid particleIndexInfoGrid,	// input, particles' indices for each cell of spatial grid.
	ParticleArray particleArray,					// input, particle array.
	VertexGrid vertexGrid,							// output, scalar field grid.
	GridInfo spatialGridInfo,						// input, spatial hasing grid information.
	GridInfo scalarGridInfo,						// input, scalar field grid information.
	SimParam params)					 
{
	// get corresponding vertex position.
	float3 vPos = getVertexPos(index1DTo3D(vertexIndex1D, vertexGrid.resolution),
		scalarGridInfo.minPos, scalarGridInfo.cellSize);
	int3 curIndex = getIndex3D(vPos, spatialGridInfo.minPos, spatialGridInfo.cellSize);

	// get influenced spatial hashing cells' bounding box and clamping.
	int3 minIndex = curIndex - 1;
	int3 maxIndex = curIndex + 1;
	minIndex = clamp(minIndex, make_int3(0, 0, 0), make_int3(particleIndexInfoGrid.resolution.x - 1,
		particleIndexInfoGrid.resolution.y - 1, particleIndexInfoGrid.resolution.z - 1));
	maxIndex = clamp(maxIndex, make_int3(0, 0, 0), make_int3(particleIndexInfoGrid.resolution.x - 1,
		particleIndexInfoGrid.resolution.y - 1, particleIndexInfoGrid.resolution.z - 1));

	//SimpleVertex* v = &vertexGrid.grid[vertexIndex1D];
	float val = 0.f;
	for (int zSp = minIndex.z; zSp <= maxIndex.z; zSp++)
	{
		for (int ySp = minIndex.y; ySp <= maxIndex.y; ySp++)
		{
			for (int xSp = minIndex.x; xSp <= maxIndex.x; xSp++)
			{
				// 3D index of spatialGrid.
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				// 粒子在particleArray中的索引信息（偏移与长度）
				IndexInfo indexInfo = particleIndexInfoGrid.grid[index3DTo1D(index3D, particleIndexInfoGrid.resolution)];
				// travel each particle of the corresponding cell to calcular scalr value.
				if (indexInfo.start == 0xffffffff)
					continue;
				for (uint i = indexInfo.start; i < indexInfo.end; i++)
				{
					float3 delta = vPos - particleArray.grid[i].pos;
					float distSq = dot(delta, delta);
					// using TC01 kernel function.
					val += kernelTC01(distSq, params.effRSq);
				}
			}
		}
	}
	vertexGrid.grid[vertexIndex1D].value = val;
}

//! func: calculate the corresponding vertex's scalar value of scalar field using ZB05 method.
__device__
void updateScalarFieldValueZB05(
	uint vertexIndex1D,								// input, vertex's index of scalar field grid.
	ParticleIndexInfoGrid particleIndexInfoGrid,	// input, particles' indices for each cell of spatial grid.
	ParticleArray particleArray,					// input, particle array.
	VertexGrid vertexGrid,							// output, scalar field grid.
	GridInfo spatialGridInfo,						// input, spatial hasing grid information.
	GridInfo scalarGridInfo,						// input, scalar field grid information.
	SimParam params)
{
	// get corresponding vertex position.
	float3 vPos = getVertexPos(index1DTo3D(vertexIndex1D, vertexGrid.resolution),
		scalarGridInfo.minPos, scalarGridInfo.cellSize);

	// get influenced spatial hashing cells' bounding box and clamping.
	int3 curIndex = getIndex3D(vPos, spatialGridInfo.minPos, spatialGridInfo.cellSize);
	int3 minIndex = curIndex - 1;
	int3 maxIndex = curIndex + 1;
	minIndex = clamp(minIndex, make_int3(0, 0, 0), make_int3(particleIndexInfoGrid.resolution.x - 1,
		particleIndexInfoGrid.resolution.y - 1, particleIndexInfoGrid.resolution.z - 1));
	maxIndex = clamp(maxIndex, make_int3(0, 0, 0), make_int3(particleIndexInfoGrid.resolution.x - 1,
		particleIndexInfoGrid.resolution.y - 1, particleIndexInfoGrid.resolution.z - 1));

	float wSum = 0.0f;
	float3 posAvg = make_float3(0.0f, 0.0f, 0.0f);
	for (int zSp = minIndex.z; zSp <= maxIndex.z; zSp++)
	{
		for (int ySp = minIndex.y; ySp <= maxIndex.y; ySp++)
		{
			for (int xSp = minIndex.x; xSp <= maxIndex.x; xSp++)
			{
				// 3D index of spatialGrid.
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexInfo indexInfo = particleIndexInfoGrid.grid[index3DTo1D(index3D, particleIndexInfoGrid.resolution)];
				// travel each particle of the corresponding cell to calcular scalr value.
				if (indexInfo.start == 0xffffffff)
					continue;
				for (uint i = indexInfo.start; i < indexInfo.end; i++)
				{
					float3 neighborPos = particleArray.grid[i].pos;
					float3 delta = vPos - neighborPos;
					float distSq = dot(delta, delta);

					// using ZB05 kernel function.
					const float wi = kernelZB05(distSq, params.effRSq);
					wSum += wi;
					posAvg += neighborPos * wi;
				}
			}
		}
	}
	if (wSum > 0.0f)
	{
		posAvg /= wSum;
		vertexGrid.grid[vertexIndex1D].value = length(vPos - posAvg) - params.particleRadius;
	}
}

//! func: call function updateVertexValue() to calculate scalar field just for surface cell.
//! Actually it's not a good idea since all the branch would be executed despite it's surface cell or not.
__global__ 
void updateScalarGridValuesStd(
	IsSurfaceGrid isSurfaceGrid,				// input, whether the corresponding grid point is in surface region or not.
	ParticleIndexInfoGrid particleIndexInfoGrid,// input, particles' indices for each cell of spatial grid.	
	ParticleArray particleArray,				// input, particles array.
	VertexGrid vertexGrid,						// output, scalar field grid.
	GridInfo spatialGridInfo,					// input, spatial hasing grid information.
	GridInfo scalarGridInfo,					// input, scalar field grid information.
	SimParam params
	)					
{
	uint threadId = getThreadIdGlobal();
	// boundary detection.
	if (threadId >= vertexGrid.size)
		return;
	// if the grid point is not in surface region, just return.
	if (isSurfaceGrid.grid[threadId] != 1)
		return;
	// call function updateVertexValue() to calculate scalar field value.
	updateScalarFieldValueTC01(threadId, particleIndexInfoGrid, particleArray,
		vertexGrid, spatialGridInfo, scalarGridInfo, params);
}

//! func: compact the surface vertices into a continuous array.(discard those that are not in surface region)
//! So we can deal with this compacted array to get higher performance without conditional branch.
__global__ 
void compactSurfaceVertex(
	SurfaceVertexIndexArray svIndexArray,	// output, compacted surface vertices' indices array.
	IsSurfaceGrid isSurfaceGridScan,		// input, exclusive prefix sum of isSurfaceGrid.
	IsSurfaceGrid isSurfaceGrid,			// input, whether the corresponding grid point is in surface region or not.
	SimParam params)			
{
	uint threadId = getThreadIdGlobal();
	if (threadId >= isSurfaceGridScan.size || isSurfaceGrid.grid[threadId] != 1)
		return;
	svIndexArray.grid[isSurfaceGridScan.grid[threadId]] = threadId;
}

//! func: call function updateVertexValue() to calculate scalar field value.
//  This time we don't need to deal with non-surface vertices.
__global__ 
void updateScalarGridValuesCompacted(
	SurfaceVertexIndexArray svIndexArray,		// input, compacted surface vertices' indices array.
	uint numSurfaceVertices,					// input, length of svIndexArray.
	ParticleIndexInfoGrid particleIndexInfoGrid,// input, particles' indices for each cell of spatial grid.	
	ParticleArray particleArray,				// input, particles' position array.
	VertexGrid vertexGrid,						// output, scalar field grid.
	GridInfo spatialGridInfo,					// input, spatial hashing grid information.
	GridInfo scalarGridInfo,					// input, scalar field grid information.
	SimParam params)					
{
	uint threadId = getThreadIdGlobal();
	if (threadId >= svIndexArray.size || threadId >= numSurfaceVertices)
		return;
	//updateScalarFieldValueTC01(svIndexArray.grid[threadId], particleIndexInfoGrid, particleArray,
	//	vertexGrid, spatialGridInfo, scalarGridInfo, params);
	updateScalarFieldValueZB05(svIndexArray.grid[threadId], particleIndexInfoGrid, particleArray,
		vertexGrid, spatialGridInfo, scalarGridInfo, params);
}

//! func: valid surface cubes detection. Here "valud" means that the cube will produce triangles.
//! We detect those valid cubes to avoid extra branch.
__global__ 
void detectValidSurfaceCubes(
	SurfaceVertexIndexArray svIndexArray,		// input, compacted surface vertices' indices array.
	uint numSurfaceVertices,					// input, length of svIndexArray.
	VertexGrid vGrid,							// input, scalar field grid.
	IsValidSurfaceGrid isValidSurfaceGrid,		// output, whether the cell is valid or not.
	NumVerticesGrid numVerticesGrid,			// output, number of vertices per cell.
	IsSurfaceGrid isSfGrid,						// input, whether the corresponding grid point is in surface region or not.
	SimParam params)						
{
	uint threadId = getThreadIdGlobal();
	if (threadId >= svIndexArray.size || threadId >= numSurfaceVertices)
		return;
	// get 3D index and boundary handling.
	uint cubeIndex1D = svIndexArray.grid[threadId];
	uint3 cubeIndex3D = index1DTo3D(cubeIndex1D, vGrid.resolution);
	if (cubeIndex3D.x >= vGrid.resolution.x - 1 ||
		cubeIndex3D.y >= vGrid.resolution.y - 1 ||
		cubeIndex3D.z >= vGrid.resolution.z - 1)
		return;
	// get 8 corners of the cube.
	uint cornerIndex1Ds[8];
	getCornerIndex1Ds(cubeIndex3D, vGrid.resolution, cornerIndex1Ds);

	// get corresponding situation flag.
	uint vertexFlag = getVertexFlag(cornerIndex1Ds, vGrid, params.isoValue);

	uint numVertices = 0;
	// 八个顶点都是表面顶点才进行三角化, 这里必须要，否则会出现双层的网格
	if (isAllSfVertex(cornerIndex1Ds, isSfGrid))
	{  
		numVertices = tex1Dfetch(numVerticesTex, vertexFlag);
	}
	
	isValidSurfaceGrid.grid[threadId] = numVertices > 0 ? 1 : 0;
	numVerticesGrid.grid[threadId] = numVertices;
}

//! func: compact the valid cubes into a continuous array.
__global__ 
void compactValidSurfaceCubes(
	ValidSurfaceIndexArray indexArray,			// output, valid cubes' indices array.
	IsValidSurfaceGrid validSurfaceGridScan,	// input, exculsive prefix sum of validSurafceGrid.
	IsValidSurfaceGrid validSurafceGrid,		// input, whether the cell is valid or not.
	SimParam params
)
{
	uint threadId = getThreadIdGlobal();
	if (threadId >= validSurfaceGridScan.size || validSurafceGrid.grid[threadId] != 1)
		return;
	// save the index of corresponding surface vertex's index.
	indexArray.grid[validSurfaceGridScan.grid[threadId]] = threadId;
}

//! func: generate triangles using marching cube algorithm.
__global__ 
void generateTriangles(
	SurfaceVertexIndexArray surfaceIndexInGridArray,// input, compacted surface vertices' indice array.
	ValidSurfaceIndexArray validIndexInSurfaceArray,// input, valid cubes' indices array.
	GridInfo scGridInfo,							// input, scalar grid information.
	NumVerticesGrid numVerticesGridScan,			// input, exculsive prefix sum of numVerticesGrid.
	VertexGrid vertexGrid,							// input, scalar field grid.
	Float3Grid posGrid,								// output, positions of triangles.
	Float3Grid norGrid,								// output, normals of triangles.
	SimParam params)								
{
	// get corresponding index and boundary handling.
	uint threadId = getThreadIdGlobal();
	if (threadId >= validIndexInSurfaceArray.size)
		return;
	// index of compacted surface vertices' indices array.
	uint surfaceIndex = validIndexInSurfaceArray.grid[threadId];
	// 1D index of sclar field grid.
	uint gridIndex = surfaceIndexInGridArray.grid[surfaceIndex];
	// 3D index of scalar field grid.
	uint3 gridIndex3D = index1DTo3D(gridIndex, vertexGrid.resolution);

	// get corresponding situation flag.
	uint vertexFlag = getVertexFlag(gridIndex3D, vertexGrid, params.isoValue);
	// get edge flag.
	uint edgeFlag = tex1Dfetch(edgeTex, vertexFlag);
	// get number of vertices.
	uint numVertices = tex1Dfetch(numVerticesTex, vertexFlag);

	// 8 corners' 3D indices of current cube.
	uint3 cornerIndex3Ds[8];
	// 8 corners' positions of current cube.
	float3 cornerPoss[8];
	// 8 corners' normals of current cube.
	float3 cornerNors[8];
	// 12 edges' intersection positions of current cube.
	float3 intersectPoss[12];
	// 12 edges' intersection normals of current cube.
	float3 intersectNormals[12];

	// get 8 corners' 3D indices.
	getCornerIndex3Ds(gridIndex3D, cornerIndex3Ds);
	// get 8 corners' positions.
	getCornerPositions(cornerIndex3Ds, scGridInfo.minPos, scGridInfo.cellSize, cornerPoss);
	// get 8 corners' normals.
	getCornerNormals(cornerIndex3Ds, vertexGrid, cornerNors);

	float sign = (params.isoValue < 0.0f) ? (-1.0f) : (1.0f);

	for (int i = 0; i < 12; i++)
	{
		// 编号为i的边与等值面相交
		if (edgeFlag & (1 << i))
		{
			uint start = tex1Dfetch(vertexIndexesOfEdgeTex, i << 1);
			uint end = tex1Dfetch(vertexIndexesOfEdgeTex, (i << 1) + 1);
			uint startIndex = index3DTo1D(cornerIndex3Ds[start], vertexGrid.resolution);
			uint endIndex = index3DTo1D(cornerIndex3Ds[end], vertexGrid.resolution);

			float startValue = vertexGrid.grid[startIndex].value;
			float endValue = vertexGrid.grid[endIndex].value;
			float lerpFac = getLerpFac(startValue, endValue, params.isoValue);
			intersectPoss[i] = lerp(cornerPoss[start], cornerPoss[end], lerpFac);
			intersectNormals[i] = sign * normalize(lerp(cornerNors[start], cornerNors[end], lerpFac));
		}
	}
	uint numTri = numVertices / 3;
	for (uint i = 0; i < numTri; i++)
	{
		for (uint j = 0; j < 3; j++)
		{
			int edgeIndex = tex1Dfetch(edgeIndexesOfTriangleTex, vertexFlag * 16 + i * 3 + j);
			uint index = numVerticesGridScan.grid[surfaceIndex] + i * 3 + j;
			posGrid.grid[index] = intersectPoss[edgeIndex];
			norGrid.grid[index] = intersectNormals[edgeIndex];
		}
	}
}

//! -----------------------------------------launch functions for cuda kernel functions----------------------------------

extern "C" 
void launchEstimateSurfaceVertices(
	dim3 gridDim_,
	dim3 blockDim_,
	DensityGrid densityGrid,
	IsSurfaceGrid isSurfaceGrid,
	uint scSearchExt,
	SimParam params)
{
	estimateSurfaceVertices << < gridDim_, blockDim_ >> > (densityGrid, isSurfaceGrid, scSearchExt, params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchUpdateScalarGridValues(
	dim3 gridDim_,
	dim3 blockDim_,
	IsSurfaceGrid isSurfaceGrid,
	ParticleIndexInfoGrid particleIndexInfoGrid,
	ParticleArray particleArray,
	VertexGrid vertexGrid,
	GridInfo spatialGridInfo,
	GridInfo scalarGridInfo,
	SimParam params)
{
	// not good enough.
	updateScalarGridValuesStd << <gridDim_, blockDim_ >> > (isSurfaceGrid, particleIndexInfoGrid,
		particleArray, vertexGrid, spatialGridInfo, scalarGridInfo, params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchCompactSurfaceVertex(
	dim3 gridDim_,
	dim3 blockDim_,
	SurfaceVertexIndexArray svIndexArray,
	IsSurfaceGrid isSurfaceGridScan,
	IsSurfaceGrid isSurfaceGrid,
	SimParam params)
{
	compactSurfaceVertex << <gridDim_, blockDim_ >> > (svIndexArray, isSurfaceGridScan, isSurfaceGrid, params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchUpdateScalarGridValuesCompacted(
	dim3 gridDim_,
	dim3 blockDim_,
	SurfaceVertexIndexArray svIndexArray,
	uint numSurfaceVertices, 
	ParticleIndexInfoGrid particleIndexInfoGrid,
	ParticleArray particleArray,
	VertexGrid vertexGrid,
	GridInfo spatialGridInfo,
	GridInfo scalarGridInfo,
	SimParam params)
{
	// extra branches are avoided.
	updateScalarGridValuesCompacted << <gridDim_, blockDim_ >> > (svIndexArray, numSurfaceVertices,
		particleIndexInfoGrid, particleArray, vertexGrid, spatialGridInfo, scalarGridInfo, params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchDetectValidSurfaceCubes(
	dim3 gridDim_,
	dim3 blockDim_,
	SurfaceVertexIndexArray svIndexArray,
	uint numSurfaceVertices,
	VertexGrid vGrid,
	IsValidSurfaceGrid isValidSurfaceGrid,
	NumVerticesGrid numVerticesGrid,
	IsSurfaceGrid isSfGrid,
	SimParam params)
{
	detectValidSurfaceCubes << <gridDim_, blockDim_ >> > (svIndexArray,
		numSurfaceVertices, vGrid, isValidSurfaceGrid, numVerticesGrid, isSfGrid, params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchCompactValidSurfaceCubes(
	dim3 gridDim_, 
	dim3 blockDim_,
	ValidSurfaceIndexArray indexArray,
	IsValidSurfaceGrid validSurfaceGridScan,
	IsValidSurfaceGrid validSurfaceGrid,
	SimParam params)
{
	compactValidSurfaceCubes << <gridDim_, blockDim_ >> > (indexArray,
		validSurfaceGridScan, validSurfaceGrid, params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchGenerateTriangles(
	dim3 gridDim_,
	dim3 blockDim_,
	SurfaceVertexIndexArray surfaceIndexInGridArray,
	ValidSurfaceIndexArray validIndexInSurfaceArray,
	GridInfo scGridInfo,
	NumVerticesGrid numVerticesGrid,
	VertexGrid vertexGrid,
	Float3Grid posGrid,
	Float3Grid norGrid,
	SimParam params)
{
	generateTriangles << <gridDim_, blockDim_ >> > (surfaceIndexInGridArray,
		validIndexInSurfaceArray, scGridInfo, numVerticesGrid, vertexGrid, posGrid, norGrid, params);
	cudaDeviceSynchronize();
}

//! --------------------------------------Spatial grid establish------------------------------------

//! func: calculation of position's corresponding grid pos.
__device__
uint3 calcGridPosKernel(float3 p, GridInfo spatialGridInfo)
{
	uint3 gridPos;
	gridPos.x = floor((p.x - spatialGridInfo.minPos.x) / spatialGridInfo.cellSize);
	gridPos.y = floor((p.y - spatialGridInfo.minPos.y) / spatialGridInfo.cellSize);
	gridPos.z = floor((p.z - spatialGridInfo.minPos.z) / spatialGridInfo.cellSize);
	return gridPos;
}

//! func: 3D index -> 1D index.
__device__
unsigned int calcGridHashKernel(int3 gridPos, GridInfo spatialGridInfo)
{
	gridPos.x = gridPos.x & (spatialGridInfo.resolution.x - 1);
	gridPos.y = gridPos.y & (spatialGridInfo.resolution.y - 1);
	gridPos.z = gridPos.z & (spatialGridInfo.resolution.z - 1);
	return gridPos.z * spatialGridInfo.resolution.x * spatialGridInfo.resolution.y
		+ gridPos.y * spatialGridInfo.resolution.x + gridPos.x;
}

//! func: calculation of particles' hash value.
__global__
void calcParticlesHashKernel(
	unsigned int *gridParticleHash,		// output, array of particles' hash value.
	ParticleArray particles,			// input, particles array.
	uint numParticles,					// input, number of particles.
	DensityGrid densityGrid,			// output, virtual density grid value.
	GridInfo spatialGridInfo)
{
	unsigned int index = getThreadIdGlobal();
	if (index >= numParticles)
		return;

	float3 curPos = particles.grid[index].pos;
	int3 gridPos = getIndex3D(curPos, spatialGridInfo.minPos, spatialGridInfo.cellSize);
	unsigned int hashValue = index3DTo1D(make_uint3(gridPos.x, gridPos.y, gridPos.z), spatialGridInfo.resolution);
	gridParticleHash[index] = hashValue;
	densityGrid.grid[hashValue] = 1.0f;
}

//! func: find start index and end index for each cell.
__global__
void findCellRangeKernel(
	ParticleIndexInfoGrid particlesIndexInforArray,	// output, each cells' start index and end index.
	uint numParticles,									// input, number of particles.
	uint *gridParticleHash)								// input, particles' hash value array.
{
	thread_block cta = this_thread_block();
	extern __shared__ unsigned int sharedHash[];
	//unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int index = getThreadIdGlobal();
	unsigned int hashValue;

	if (index < numParticles)
	{
		hashValue = gridParticleHash[index];
		sharedHash[threadIdx.x + 1] = hashValue;

		// first thread in block must load neighbor particle hash
		if (index > 0 && threadIdx.x == 0)
			sharedHash[0] = gridParticleHash[index - 1];
	}

	sync(cta);
	
	if (index < numParticles)
	{
		if (index == 0 || hashValue != sharedHash[threadIdx.x])
		{
			particlesIndexInforArray.grid[hashValue].start = index;
			if (index > 0)
				particlesIndexInforArray.grid[sharedHash[threadIdx.x]].end = index;
		}

		if (index == numParticles - 1)
			particlesIndexInforArray.grid[hashValue].end = index + 1;
	}
}

void launchSpatialGridBuilding(
	ParticleArray *particlesArray,
	uint numParticles,
	ParticleIndexInfoGrid *particlesIndexInforArray,
	DensityGrid *densityGrid,
	GridInfo spatialGridInfo)
{
	//! memory allocation for particles' hash value's storage.
	uint *dGridParticleHash;
	cudaMalloc((void**)&dGridParticleHash, numParticles * sizeof(unsigned int));

	//! calculation of grid dim and block dim.
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

	//! step1: computation of hash value of particles.
	calcParticlesHashKernel << <numBlocks, numThreads >> > (
		dGridParticleHash, *particlesArray, numParticles, *densityGrid, spatialGridInfo);
	getLastCudaError("calcParticlesHashKernel");
	cudaDeviceSynchronize();

	//! step2: sort the particle according to their hash value.
	thrust::sort_by_key(
		thrust::device_ptr<unsigned int>(dGridParticleHash),
		thrust::device_ptr<unsigned int>(dGridParticleHash + numParticles),
		thrust::device_ptr<SimpleParticle>(particlesArray->grid));
	getLastCudaError("sort_by_key");
	cudaDeviceSynchronize();

	//! step3: find start index and end index of each cell.
	// 0xffffffff, need to be attentioned.
	cudaMemset(particlesIndexInforArray->grid, 0xffffffff, particlesIndexInforArray->size * sizeof(IndexInfo));
	unsigned int memSize = sizeof(unsigned int) * (numThreads + 1);
	findCellRangeKernel << < numBlocks, numThreads, memSize >> > (*particlesIndexInforArray, numParticles,
		dGridParticleHash);
	getLastCudaError("findCellRangeKernel");
	cudaDeviceSynchronize();

	//! freee memory.
	cudaFree(dGridParticleHash);
}
