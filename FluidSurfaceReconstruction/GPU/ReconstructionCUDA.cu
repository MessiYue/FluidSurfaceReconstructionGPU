
#include <Windows.h>
#include "ReconstructionCUDA.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <cooperative_groups.h>
#include <thrust/iterator/zip_iterator.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

#include "Defines.h"
#include "CudaUtils.h"
#include "SVD3.cuh"

using namespace cooperative_groups;

__constant__ float EPSILON_ = (float)1.0e-7;
__constant__ float SIGMA = 315.0f / (64.0f * 3.1415926535898f);

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

//! func: extraction of surface particles.
__global__
void extractionOfSurfaceParticles(
	SimParam simParam,
	DensityGrid flagGrid,
	IsSurfaceGrid surfaceParticlesFlagGrid,
	ParticleIndexRangeGrid particleIndexRangeGrid)
{
	uint threadId = getThreadIdGlobal();
	uint3 flagGridRes = flagGrid.resolution;
	uint3 curIndex3D = index1DTo3D(threadId, particleIndexRangeGrid.resolution);

	if (curIndex3D.x >= flagGridRes.x - 1 || curIndex3D.y >= flagGridRes.y - 1 || curIndex3D.z >= flagGridRes.z - 1)
		return;

	if (flagGrid.grid[threadId] < 1.0f)
		return;

	bool isSurface = false;
	int3 lower = make_int3(curIndex3D.x - 1, curIndex3D.y - 1, curIndex3D.z - 1);
	int3 upper = make_int3(curIndex3D.x + 1, curIndex3D.y + 1, curIndex3D.z + 1);
	lower = clamp(lower, make_int3(0, 0, 0), make_int3(flagGridRes.x - 1, flagGridRes.y - 1, flagGridRes.z - 1));
	upper = clamp(upper, make_int3(0, 0, 0), make_int3(flagGridRes.x - 1, flagGridRes.y - 1, flagGridRes.z - 1));
#pragma unroll 3
	for (int z = lower.z; z <= upper.z; ++z)
	{
#pragma unroll 3
		for (int y = lower.y; y <= upper.y; ++y)
		{
#pragma unroll 3
			for (int x = lower.x; x <= upper.x; ++x)
			{
				uint3 neighbor = make_uint3(x, y, z);
				uint index = index3DTo1D(neighbor, flagGridRes);
				float value = flagGrid.grid[index];
				if (value < 1.0f)
					isSurface = true;
			}
		}
	}

	if (!isSurface)
		return;

	//! mark particles in this cell as surface particles.
	IndexRange indexRange = particleIndexRangeGrid.grid[threadId];
	if (indexRange.start != 0xffffffff)
	{
		for (uint i = indexRange.start; i < indexRange.end; ++i)
			surfaceParticlesFlagGrid.grid[i] = 1;
	}
}

//! func: surface vertices' estimation.
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
	ParticleIndexRangeGrid particleIndexRangeGrid,	// input, particles' indices for each cell of spatial grid.
	ParticleArray particleArray,					// input, particle array.
	ScalarFieldGrid ScalarFieldGrid,							// output, scalar field grid.
	GridInfo spatialGridInfo,						// input, spatial hasing grid information.
	GridInfo scalarGridInfo,						// input, scalar field grid information.
	SimParam params)					 
{
	// get corresponding vertex position.
	float3 vPos = getVertexPos(index1DTo3D(vertexIndex1D, ScalarFieldGrid.resolution),
		scalarGridInfo.minPos, scalarGridInfo.cellSize);
	int3 curIndex = getIndex3D(vPos, spatialGridInfo.minPos, spatialGridInfo.cellSize);

	// get influenced spatial hashing cells' bounding box and clamping.
	int3 minIndex = curIndex - 1;
	int3 maxIndex = curIndex + 1;
	minIndex = clamp(minIndex, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));
	maxIndex = clamp(maxIndex, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));

	//ScalarValue* v = &ScalarFieldGrid.grid[vertexIndex1D];
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
				IndexRange IndexRange = particleIndexRangeGrid.grid[index3DTo1D(index3D, particleIndexRangeGrid.resolution)];
				// travel each particle of the corresponding cell to calcular scalr value.
				if (IndexRange.start == 0xffffffff)
					continue;
				for (uint i = IndexRange.start; i < IndexRange.end; i++)
				{
					float3 delta = vPos - particleArray.grid[i].pos;
					float distSq = dot(delta, delta);
					// using TC01 kernel function.
					val += kernelTC01(distSq, params.smoothingRadiusSq);
				}
			}
		}
	}
	ScalarFieldGrid.grid[vertexIndex1D].value = val;
}

//! func: calculate the corresponding vertex's scalar value of scalar field using ZB05 method.
__device__
void updateScalarFieldValueZB05(
	uint vertexIndex1D,								// input, vertex's index of scalar field grid.
	ParticleIndexRangeGrid particleIndexRangeGrid,	// input, particles' indices for each cell of spatial grid.
	ParticleArray particleArray,					// input, particle array.
	ScalarFieldGrid ScalarFieldGrid,							// output, scalar field grid.
	GridInfo spatialGridInfo,						// input, spatial hasing grid information.
	GridInfo scalarGridInfo,						// input, scalar field grid information.
	SimParam params)
{
	// get corresponding vertex position.
	float3 vPos = getVertexPos(index1DTo3D(vertexIndex1D, ScalarFieldGrid.resolution),
		scalarGridInfo.minPos, scalarGridInfo.cellSize);

	// get influenced spatial hashing cells' bounding box and clamping.
	int3 curIndex = getIndex3D(vPos, spatialGridInfo.minPos, spatialGridInfo.cellSize);
	int3 minIndex = curIndex - 1;
	int3 maxIndex = curIndex + 1;
	minIndex = clamp(minIndex, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));
	maxIndex = clamp(maxIndex, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));

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
				IndexRange IndexRange = particleIndexRangeGrid.grid[index3DTo1D(index3D, particleIndexRangeGrid.resolution)];
				// travel each particle of the corresponding cell to calcular scalr value.
				if (IndexRange.start == 0xffffffff)
					continue;
				for (uint i = IndexRange.start; i < IndexRange.end; i++)
				{
					float3 neighborPos = particleArray.grid[i].pos;
					float3 delta = vPos - neighborPos;
					float distSq = dot(delta, delta);

					// using ZB05 kernel function.
					const float wi = kernelZB05(distSq, params.smoothingRadiusSq);
					wSum += wi;
					posAvg += neighborPos * wi;
				}
			}
		}
	}
	if (wSum > 0.0f)
	{
		posAvg /= wSum;
		ScalarFieldGrid.grid[vertexIndex1D].value = length(vPos - posAvg) - params.particleRadius;
	}
}

//! func: call function updateVertexValue() to calculate scalar field just for surface cell.
//! Actually it's not a good idea since all the branch would be executed despite it's surface cell or not.
__global__ 
void updateScalarGridValuesStd(
	IsSurfaceGrid isSurfaceGrid,				// input, whether the corresponding grid point is in surface region or not.
	ParticleIndexRangeGrid particleIndexRangeGrid,// input, particles' indices for each cell of spatial grid.	
	ParticleArray particleArray,				// input, particles array.
	ScalarFieldGrid ScalarFieldGrid,						// output, scalar field grid.
	GridInfo spatialGridInfo,					// input, spatial hasing grid information.
	GridInfo scalarGridInfo,					// input, scalar field grid information.
	SimParam params
	)					
{
	uint threadId = getThreadIdGlobal();
	// boundary detection.
	if (threadId >= ScalarFieldGrid.size)
		return;
	// if the grid point is not in surface region, just return.
	if (isSurfaceGrid.grid[threadId] != 1)
		return;
	// call function updateVertexValue() to calculate scalar field value.
	updateScalarFieldValueTC01(threadId, particleIndexRangeGrid, particleArray,
		ScalarFieldGrid, spatialGridInfo, scalarGridInfo, params);
}

//! func: compact the surface vertices into a continuous array.(discard those that are not in surface region)
//! So we can deal with this compacted array to get higher performance without conditional branch.
__global__ 
void compactSurfaceVertex(
	SurfaceVerticesIndexArray svIndexArray,	// output, compacted surface vertices' indices array.
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
	SurfaceVerticesIndexArray svIndexArray,		// input, compacted surface vertices' indices array.
	uint numSurfaceVertices,					// input, length of svIndexArray.
	ParticleIndexRangeGrid particleIndexRangeGrid,// input, particles' indices for each cell of spatial grid.	
	ParticleArray particleArray,				// input, particles' position array.
	ScalarFieldGrid ScalarFieldGrid,						// output, scalar field grid.
	GridInfo spatialGridInfo,					// input, spatial hashing grid information.
	GridInfo scalarGridInfo,					// input, scalar field grid information.
	SimParam params)					
{
	uint threadId = getThreadIdGlobal();
	if (threadId >= svIndexArray.size || threadId >= numSurfaceVertices)
		return;
	//updateScalarFieldValueTC01(svIndexArray.grid[threadId], particleIndexRangeGrid, particleArray,
	//	ScalarFieldGrid, spatialGridInfo, scalarGridInfo, params);
	updateScalarFieldValueZB05(svIndexArray.grid[threadId], particleIndexRangeGrid, particleArray,
		ScalarFieldGrid, spatialGridInfo, scalarGridInfo, params);
}

//! func: valid surface cubes detection. Here "valud" means that the cube will produce triangles.
//! We detect those valid cubes to avoid extra branch.
__global__ 
void detectValidSurfaceCubes(
	SurfaceVerticesIndexArray svIndexArray,		// input, compacted surface vertices' indices array.
	uint numSurfaceVertices,					// input, length of svIndexArray.
	ScalarFieldGrid vGrid,						// input, scalar field grid.
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
	SurfaceVerticesIndexArray surfaceIndexInGridArray,// input, compacted surface vertices' indice array.
	ValidSurfaceIndexArray validIndexInSurfaceArray,// input, valid cubes' indices array.
	GridInfo scGridInfo,							// input, scalar grid information.
	NumVerticesGrid numVerticesGridScan,			// input, exculsive prefix sum of numVerticesGrid.
	ScalarFieldGrid ScalarFieldGrid,							// input, scalar field grid.
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
	uint3 gridIndex3D = index1DTo3D(gridIndex, ScalarFieldGrid.resolution);

	// get corresponding situation flag.
	uint vertexFlag = getVertexFlag(gridIndex3D, ScalarFieldGrid, params.isoValue);
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
	getCornerNormals(cornerIndex3Ds, ScalarFieldGrid, cornerNors);

	float sign = (params.isoValue < 0.0f) ? (-1.0f) : (1.0f);

	for (int i = 0; i < 12; i++)
	{
		// 编号为i的边与等值面相交
		if (edgeFlag & (1 << i))
		{
			uint start = tex1Dfetch(vertexIndexesOfEdgeTex, i << 1);
			uint end = tex1Dfetch(vertexIndexesOfEdgeTex, (i << 1) + 1);
			uint startIndex = index3DTo1D(cornerIndex3Ds[start], ScalarFieldGrid.resolution);
			uint endIndex = index3DTo1D(cornerIndex3Ds[end], ScalarFieldGrid.resolution);

			float startValue = ScalarFieldGrid.grid[startIndex].value;
			float endValue = ScalarFieldGrid.grid[endIndex].value;
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

//! extraction of surface and involve particles.
__global__
void extractionOfSurfaceAndInvolveParticles(
	SimParam simParam,									// input, simulation parameter.
	DensityGrid flagGrid,								// input, flag grid.
	NumInvolveParticlesGrid numInvolveParticlesGrid,	// output, number of involved particles per spatial cell.
	IsSurfaceGrid surfaceParticlesFlagGrid,				// output, surface particles' flag array.
	ParticleIndexRangeGrid particleIndexRangeGrid)		// input, paritcle index information grid.
{
	uint threadId = getThreadIdGlobal();
	uint3 flagGridRes = flagGrid.resolution;
	uint3 curIndex3D = index1DTo3D(threadId, particleIndexRangeGrid.resolution);

	if (curIndex3D.x >= flagGridRes.x - 1 || curIndex3D.y >= flagGridRes.y - 1 || curIndex3D.z >= flagGridRes.z - 1)
		return;

	if (flagGrid.grid[threadId] < 1.0f)
		return;

	bool isSurface = false;
	int3 lower = make_int3(curIndex3D.x - 1, curIndex3D.y - 1, curIndex3D.z - 1);
	int3 upper = make_int3(curIndex3D.x + 1, curIndex3D.y + 1, curIndex3D.z + 1);
	lower = clamp(lower, make_int3(0, 0, 0), make_int3(flagGridRes.x - 1, flagGridRes.y - 1, flagGridRes.z - 1));
	upper = clamp(upper, make_int3(0, 0, 0), make_int3(flagGridRes.x - 1, flagGridRes.y - 1, flagGridRes.z - 1));
#pragma unroll 3
	for (int z = lower.z; z <= upper.z; ++z)
	{
#pragma unroll 3
		for (int y = lower.y; y <= upper.y; ++y)
		{
#pragma unroll 3
			for (int x = lower.x; x <= upper.x; ++x)
			{
				uint3 neighbor = make_uint3(x, y, z);
				uint index = index3DTo1D(neighbor, flagGridRes);
				float value = flagGrid.grid[index];
				if (value < 1.0f)
					isSurface = true;
			}
		}
	}

	if (!isSurface)
		return;

	//! mark particles in this cell as surface particles.
	IndexRange indexRange = particleIndexRangeGrid.grid[threadId];
	uint count = indexRange.end - indexRange.start;
	if (indexRange.start != 0xffffffff)
	{
		for (uint i = indexRange.start; i < indexRange.end; ++i)
			surfaceParticlesFlagGrid.grid[i] = 1;
	}

	//! collection of involve particles.
	int3 minSpatialIndex3D = make_int3(curIndex3D.x - 2, curIndex3D.y - 2, curIndex3D.z - 2);
	int3 maxSpatialIndex3D = make_int3(curIndex3D.x + 2, curIndex3D.y + 2, curIndex3D.z + 2);
	minSpatialIndex3D = clamp(minSpatialIndex3D, make_int3(0, 0, 0),
		make_int3(
			particleIndexRangeGrid.resolution.x - 1,
			particleIndexRangeGrid.resolution.y - 1,
			particleIndexRangeGrid.resolution.z - 1));
	maxSpatialIndex3D = clamp(maxSpatialIndex3D, make_int3(0, 0, 0),
		make_int3(
			particleIndexRangeGrid.resolution.x - 1,
			particleIndexRangeGrid.resolution.y - 1,
			particleIndexRangeGrid.resolution.z - 1));
	for (int zSp = minSpatialIndex3D.z; zSp <= maxSpatialIndex3D.z; ++zSp)
	{
		for (int ySp = minSpatialIndex3D.y; ySp <= maxSpatialIndex3D.y; ++ySp)
		{
			for (int xSp = minSpatialIndex3D.x; xSp <= maxSpatialIndex3D.x; ++xSp)
			{
				// 3D index of spatialGrid.
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				uint index1D = index3DTo1D(index3D, particleIndexRangeGrid.resolution);
				IndexRange indexRange = particleIndexRangeGrid.grid[index1D];
				// travel each particle of the corresponding cell to calculate scalr value.
				if (indexRange.start == 0xffffffff)
					continue;
				// assign the number of particles in that cell.
				uint num = indexRange.end - indexRange.start;
				numInvolveParticlesGrid.grid[index1D] = num;
			}
		}
	}
}

//! func: estimation of surface vertices and particles.
__global__
void estimationOfSurfaceVertices(
	SimParam simParam,									// input, simulation parameter.
	GridInfo scalarGridInfo,							// input, scalar field grid information.
	ParticleArray particleArray,						// input, particles array.
	IsSurfaceGrid surfaceParticlesFlagGrid,				// input, surface particles' flag array.
	ParticleIndexRangeGrid particleIndexRangeGrid,		// input, paritcle index information grid.
	IsSurfaceGrid isSurfaceGrid)						// output, surface vertices' flag array.
{
	uint threadId = getThreadIdGlobal();
	if (threadId >= surfaceParticlesFlagGrid.size || surfaceParticlesFlagGrid.grid[threadId] == 0)
		return;

	float3 pos = particleArray.grid[threadId].pos;
	int3 index3D = getIndex3D(pos, scalarGridInfo.minPos, scalarGridInfo.cellSize);
	uint extent = /*simParam.smoothingRadius / scalarGridInfo.cellSize + 1*/3;
	
	//! mark the vertex inside the bounding box as surface vertex.
	int3 minIndex3D = index3D - extent;
	int3 maxIndex3D = index3D + extent;
	minIndex3D = clamp(minIndex3D, make_int3(0, 0, 0), make_int3(isSurfaceGrid.resolution.x - 1,
		isSurfaceGrid.resolution.y - 1, isSurfaceGrid.resolution.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_int3(0, 0, 0), make_int3(isSurfaceGrid.resolution.x - 1,
		isSurfaceGrid.resolution.y - 1, isSurfaceGrid.resolution.z - 1));
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; ++zSp)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ++ySp)
		{
			for (int xSp = minIndex3D.x; xSp < maxIndex3D.x; ++xSp)
			{
				uint3 curIndex3D = make_uint3(xSp, ySp, zSp);
				float3 vertex = getVertexPos(curIndex3D, scalarGridInfo.minPos, scalarGridInfo.cellSize);
				float dist = length(vertex - pos);
				if (dist < simParam.smoothingRadius * 1.5)
				{
					uint curIndex1D = index3DTo1D(curIndex3D, isSurfaceGrid.resolution);
					isSurfaceGrid.grid[curIndex1D] = 1;
				}
			}
		}
	}
}

//! calculation of mean and smoothed particles.
__global__
void calculationOfMeanAndSmoothParticles(
	SimParam simParam,									// input, simulation parameter.
	ParticleArray particleArray,						// input, particles array.
	ParticleArray meanParticleArray,					// output, mean postions of particles.
	ParticleArray smoothedParticleArray,				// output, smoothed positions of particles.
	ParticleIndexRangeGrid particleIndexRangeGrid,		// input, paritcle index information grid.
	GridInfo spatialGridInfo)							// input, spatial hashing grid information.
{
	//! get corresponding index and boundary handling.
	uint threadId = getThreadIdGlobal();
	if (threadId > particleArray.size)
		return;

	//! corresponding particle position. 
	float3 pos = particleArray.grid[threadId].pos;
	int3 cubeIndex3D = getIndex3D(pos, spatialGridInfo.minPos, spatialGridInfo.cellSize);
	int3 minIndex3D = cubeIndex3D - 2;
	int3 maxIndex3D = cubeIndex3D + 2;
	minIndex3D = clamp(minIndex3D, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));

	float wSum = 0.0f;
	float3 posMean = make_float3(0.0f, 0.0f, 0.0f);
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				// 3D index of spatialGrid.
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particleIndexRangeGrid.grid[index3DTo1D(index3D,
					particleIndexRangeGrid.resolution)];
				// travel each particle of the corresponding cell to calculate scalr value.
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particleArray.grid[i].pos;
					float3 delta = pos - neighborPos;
					float dist = sqrt(dot(delta, delta));

					// using ZB05 kernel function.
					const float wj = wij(dist, simParam.anisotropicRadius);
					wSum += wj;
					posMean += neighborPos * wj;
				}
			}
		}
	}
	if (wSum > 0.0f)
	{
		posMean /= wSum;
		meanParticleArray.grid[threadId].pos = posMean;
		smoothedParticleArray.grid[threadId].pos =
			(1.0 - simParam.lambdaForSmoothed) * pos + simParam.lambdaForSmoothed * posMean;
	}
}

//! func: compactation of surface vertices and particles.
__global__
void compactationOfSurfaceVerticesAndParticles(
	SimParam simParam,										// input, simulation parameters.
	IsSurfaceGrid isSurfaceGrid,							// input, surface tag for scalar grid.
	IsSurfaceGrid isSurfaceGridScan,						// input, exclusive prefix sum of isSurfaceGrid.
	NumInvolveParticlesGrid numInvolveParticlesGrid,		// input, number of surface particles for each spatial cell.
	NumInvolveParticlesGrid numInvolveParticlesGridScan,	// input, exclusive prefix sum of NumInvolveParticlesGrid.
	ParticleIndexRangeGrid particleIndexRangeGrid,			// input, each spatial cell's particles' indices.
	SurfaceVerticesIndexArray surfaceVerticesIndexArray,	// output, compacted surface vertices' indices
	SurfaceParticlesIndexArray surfaceParticlesIndexArray)	// output, compacted surface particles' indices.
{
	uint threadId = getThreadIdGlobal();
	//! save surface vertices' indices to surfaceVerticesIndexArray.
	if (threadId < isSurfaceGridScan.size && isSurfaceGrid.grid[threadId] == 1)
		surfaceVerticesIndexArray.grid[isSurfaceGridScan.grid[threadId]] = threadId;

	//! save surface particles' indices to surfaceParticlesIndexArray.
	if (threadId < numInvolveParticlesGridScan.size && numInvolveParticlesGrid.grid[threadId] > 0)
	{
		uint start = numInvolveParticlesGridScan.grid[threadId];
		IndexRange range = particleIndexRangeGrid.grid[threadId];
		uint count = range.end - range.start;
		for (uint i = 0; i < count; ++i)
			surfaceParticlesIndexArray.grid[start + i] = range.start + i;
	}
}

//! func: calculation of transform matrices for each surface particle.
__global__
void calculationOfTransformMatricesForParticles(
	SimParam simParam,										// input, simulation parameters.
	GridInfo spatialGridInfo,								// input, spatial hashing grid information.
	ParticleArray meanParticleArray,						// input, mean positions of particles.
	ParticleArray particleArray,							// input, original positions of particles.
	ParticleIndexRangeGrid particleIndexRangeGrid,			// input, each spatial cell's particles' indices' range.
	SurfaceParticlesIndexArray surfaceParticlesIndexArray,	// input, surface particles' indices.
	MatrixArray svdMatricesArray)							// output, G matrix for surface particles.
{
	uint threadId = getThreadIdGlobal();
	
	if (threadId >= surfaceParticlesIndexArray.size)
		return;

	uint particleIndex = surfaceParticlesIndexArray.grid[threadId];
	
	//! original position and mean position.
	float3 pos = particleArray.grid[particleIndex].pos;
	float3 posMean = meanParticleArray.grid[particleIndex].pos;

	int3 cubeIndex3D = getIndex3D(pos, spatialGridInfo.minPos, spatialGridInfo.cellSize);
	int3 minIndex3D = cubeIndex3D - 2;
	int3 maxIndex3D = cubeIndex3D + 2;
	minIndex3D = clamp(minIndex3D, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));
	
	// We start with small scale matrix (h*h) in order to
	// prevent zero covariance matrix when points are all
	// perfectly lined up.
	MatrixValue cov;
	cov.a11 = cov.a22 = cov.a33 = simParam.smoothingRadiusSq;
	cov.a12 = cov.a13 = cov.a21 = cov.a23 = cov.a31 = cov.a32 = 0;

	//! computation of covariance matrix.
	uint numNeighbors = 0;
	float wSum = 0.0f;
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				// 3D index of spatialGrid.
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particleIndexRangeGrid.grid[index3DTo1D(index3D,
					particleIndexRangeGrid.resolution)];
				// travel each particle of the corresponding cell to calculate scalr value.
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particleArray.grid[i].pos;
					float3 v = neighborPos - posMean;
					float dist = sqrt(dot(v, v));
					if (dist <= simParam.anisotropicRadius)
						++numNeighbors;
					const float wj = wij(dist, simParam.anisotropicRadius);
					wSum += wj;
					
					cov.a11 += wj * v.x * v.x;
					cov.a22 += wj * v.y * v.y;
					cov.a33 += wj * v.z * v.z;
					float c_12_21 = wj * v.x * v.y;
					float c_13_31 = wj * v.x * v.z;
					float c_23_32 = wj * v.y * v.z;
					cov.a12 += c_12_21;
					cov.a21 += c_12_21;
					cov.a13 += c_13_31;
					cov.a31 += c_13_31;
					cov.a23 += c_23_32;
					cov.a32 += c_23_32;
				}
			}
		}
	}

	MatrixValue ret;
	if (numNeighbors < simParam.minNumNeighbors)
	{
		//! isolated particle.
		ret.a11 = ret.a22 = ret.a33 = simParam.smoothingRadiusInv;
		ret.a12 = ret.a13 = ret.a21 = ret.a23 = ret.a31 = ret.a32 = 0;
	}
	else
	{
		cov.a11 /= wSum; cov.a12 /= wSum; cov.a13 /= wSum;
		cov.a21 /= wSum; cov.a22 /= wSum; cov.a23 /= wSum;
		cov.a31 /= wSum; cov.a32 /= wSum; cov.a33 /= wSum;
		
		//! singular value decomposition.
		MatrixValue u;
		float3 v;
		MatrixValue w;
		svd(cov.a11, cov.a12, cov.a13, cov.a21, cov.a22, cov.a23, cov.a31, cov.a32, cov.a33,
			u.a11, u.a12, u.a13, u.a21, u.a22, u.a23, u.a31, u.a32, u.a33,
			v.x, v.y, v.z,
			w.a11, w.a12, w.a13, w.a21, w.a22, w.a23, w.a31, w.a32, w.a33);
		
		//! take off the sign
		v.x = fabsf(v.x);
		v.y = fabsf(v.y);
		v.z = fabsf(v.z);

		//! constrain Sigma
		float maxSingularVal = max(v.x, max(v.y, v.z)) / 4.0f;
		v.x = max(v.x, maxSingularVal);
		v.y = max(v.y, maxSingularVal);
		v.z = max(v.z, maxSingularVal);

		//! (invSigma * u.tranposed).
		float3 invV;
		invV.x = 1.0f / v.x;
		invV.y = 1.0f / v.y;
		invV.z = 1.0f / v.z;
		ret.a11 = u.a11 * invV.x; ret.a12 = u.a21 * invV.x; ret.a13 = u.a31 * invV.x;
		ret.a21 = u.a12 * invV.y; ret.a22 = u.a22 * invV.y; ret.a23 = u.a32 * invV.y;
		ret.a31 = u.a13 * invV.z; ret.a32 = u.a23 * invV.z; ret.a33 = u.a33 * invV.z;

		//! w * (invSigma * u.tranposed).
		matrixMul(w.a11, w.a12, w.a13, w.a21, w.a22, w.a23, w.a31, w.a32, w.a33,
			ret.a11, ret.a12, ret.a13, ret.a21, ret.a22, ret.a23, ret.a31, ret.a32, ret.a33,
			ret.a11, ret.a12, ret.a13, ret.a21, ret.a22, ret.a23, ret.a31, ret.a32, ret.a33);

		//! scaling for volume preservation
		float scale = powf(v.x * v.y * v.z, 1.0 / 3.0);
		float cof = simParam.smoothingRadiusInv * scale;
		
		//! finally we get matrix G.
		ret.a11 *= cof; ret.a12 *= cof; ret.a13 *= cof;
		ret.a21 *= cof; ret.a22 *= cof; ret.a23 *= cof;
		ret.a31 *= cof; ret.a32 *= cof; ret.a33 *= cof;
	}

	svdMatricesArray.grid[threadId] = ret;

}

//! func: computation of scalar field grid using anisotropic kernel.
__global__
void computationOfScalarFieldGrid(
	SimParam simParam,									// input, simulation parameters.
	uint numSurfaceVertices,							// input, number of surface vertices.
	GridInfo scalarGridInfo,							// input, scalar field grid information.
	GridInfo spatialGridInfo,							// input, spatial hashing grid information.
	MatrixArray svdMatricesArray,						// input, transform matrices for surface particles.
	ParticleArray smoothedParticleArray,				// input, smoothed positions of particles.
	ScalarFieldGrid particlesDensityArray,				// input, particle densities array. 
	ParticleIndexRangeGrid particleIndexRangeGrid,		// input, each spatial cell's particles' indices' range.
	SurfaceVerticesIndexArray surfaceVerticesIndexArray,// input, surface vertices' indices.
	NumInvolveParticlesGrid numInvolveParticlesGrid,	// input, number of surface particles for each cell.
	NumInvolveParticlesGrid numInvolveParticlesGridScan,// input, exclusive prefix sum of NumInvolveParticlesGrid. 
	ScalarFieldGrid scalarFieldGrid)					// output, scalar field grid.
{
	uint threadId = getThreadIdGlobal();
	if (threadId >= surfaceVerticesIndexArray.size || threadId >= numSurfaceVertices)
		return;

	uint svIndex = surfaceVerticesIndexArray.grid[threadId];

	// get corresponding vertex position.
	float3 vPos = getVertexPos(index1DTo3D(svIndex, scalarFieldGrid.resolution), 
		scalarGridInfo.minPos, scalarGridInfo.cellSize);

	// get influenced spatial hashing cells' bounding box and clamping.
	int3 curIndex3D = getIndex3D(vPos, spatialGridInfo.minPos, spatialGridInfo.cellSize);
	int3 minIndex3D = curIndex3D - 2;
	int3 maxIndex3D = curIndex3D + 2;
	minIndex3D = clamp(minIndex3D, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_int3(0, 0, 0), make_int3(particleIndexRangeGrid.resolution.x - 1,
		particleIndexRangeGrid.resolution.y - 1, particleIndexRangeGrid.resolution.z - 1));

	float sum = 0.0f;
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				// 3D index of spatialGrid.
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				uint index1D = index3DTo1D(index3D, particleIndexRangeGrid.resolution);
				// for near surface particles.
				if (numInvolveParticlesGrid.grid[index1D] > 0)
				{
					uint start = numInvolveParticlesGridScan.grid[index1D];
					IndexRange range = particleIndexRangeGrid.grid[index1D];
					uint count = range.end - range.start;
					for (uint i = 0; i < count; ++i)
					{
						MatrixValue gMat = svdMatricesArray.grid[start + i];
						float3 neighborPos = smoothedParticleArray.grid[range.start + i].pos;
						sum += simParam.particleMass / particlesDensityArray.grid[range.start + i].value
							* anisotropicW(neighborPos - vPos, gMat, determinant(gMat));
					}
				}
				else
				{
					IndexRange range = particleIndexRangeGrid.grid[index1D];
					if (range.start == 0xffffffff)
						continue;
					uint count = range.end - range.start;
					for (uint i = 0; i < count; ++i)
					{
						float3 neighborPos = smoothedParticleArray.grid[range.start + i].pos;
						sum += simParam.particleMass / particlesDensityArray.grid[range.start + i].value
							* anisotropicWOpt(neighborPos - vPos);
					}
				}
			}
		}
	}

	scalarFieldGrid.grid[svIndex].value = sum - 0.5f;
}

//! -----------------------------------------launch functions for cuda kernel functions----------------------------------

extern "C"
void launchExtractionOfSurfaceParticles(
	dim3 gridDim_,
	dim3 blockDim_,
	SimParam simParam,
	DensityGrid flagGrid,
	IsSurfaceGrid surfaceParticlesFlagGrid,
	ParticleIndexRangeGrid particleIndexRangeGrid)
{
	extractionOfSurfaceParticles << <gridDim_, blockDim_ >> > (
		simParam,
		flagGrid,
		surfaceParticlesFlagGrid,
		particleIndexRangeGrid);
	cudaDeviceSynchronize();
}

extern "C" 
void launchEstimateSurfaceVertices(
	dim3 gridDim_,
	dim3 blockDim_,
	DensityGrid densityGrid,
	IsSurfaceGrid isSurfaceGrid,
	uint scSearchExt,
	SimParam params)
{
	estimateSurfaceVertices << < gridDim_, blockDim_ >> > (
		densityGrid,
		isSurfaceGrid,
		scSearchExt,
		params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchUpdateScalarGridValues(
	dim3 gridDim_,
	dim3 blockDim_,
	IsSurfaceGrid isSurfaceGrid,
	ParticleIndexRangeGrid particleIndexRangeGrid,
	ParticleArray particleArray,
	ScalarFieldGrid ScalarFieldGrid,
	GridInfo spatialGridInfo,
	GridInfo scalarGridInfo,
	SimParam params)
{
	// not good enough.
	updateScalarGridValuesStd << <gridDim_, blockDim_ >> > (
		isSurfaceGrid,
		particleIndexRangeGrid,
		particleArray,
		ScalarFieldGrid,
		spatialGridInfo,
		scalarGridInfo,
		params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchCompactSurfaceVertex(
	dim3 gridDim_,
	dim3 blockDim_,
	SurfaceVerticesIndexArray svIndexArray,
	IsSurfaceGrid isSurfaceGridScan,
	IsSurfaceGrid isSurfaceGrid,
	SimParam params)
{
	compactSurfaceVertex << <gridDim_, blockDim_ >> > (
		svIndexArray,
		isSurfaceGridScan,
		isSurfaceGrid,
		params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchUpdateScalarGridValuesCompacted(
	dim3 gridDim_,
	dim3 blockDim_,
	SurfaceVerticesIndexArray svIndexArray,
	uint numSurfaceVertices, 
	ParticleIndexRangeGrid particleIndexRangeGrid,
	ParticleArray particleArray,
	ScalarFieldGrid ScalarFieldGrid,
	GridInfo spatialGridInfo,
	GridInfo scalarGridInfo,
	SimParam params)
{
	// extra branches are avoided.
	updateScalarGridValuesCompacted << <gridDim_, blockDim_ >> > (
		svIndexArray,
		numSurfaceVertices,
		particleIndexRangeGrid,
		particleArray,
		ScalarFieldGrid,
		spatialGridInfo,
		scalarGridInfo,
		params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchDetectValidSurfaceCubes(
	dim3 gridDim_,
	dim3 blockDim_,
	SurfaceVerticesIndexArray svIndexArray,
	uint numSurfaceVertices,
	ScalarFieldGrid vGrid,
	IsValidSurfaceGrid isValidSurfaceGrid,
	NumVerticesGrid numVerticesGrid,
	IsSurfaceGrid isSfGrid,
	SimParam params)
{
	detectValidSurfaceCubes << <gridDim_, blockDim_ >> > (
		svIndexArray,
		numSurfaceVertices,
		vGrid,
		isValidSurfaceGrid,
		numVerticesGrid,
		isSfGrid,
		params);
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
	compactValidSurfaceCubes << <gridDim_, blockDim_ >> > (
		indexArray,
		validSurfaceGridScan,
		validSurfaceGrid, 
		params);
	cudaDeviceSynchronize();
}

extern "C" 
void launchGenerateTriangles(
	dim3 gridDim_,
	dim3 blockDim_,
	SurfaceVerticesIndexArray surfaceIndexInGridArray,
	ValidSurfaceIndexArray validIndexInSurfaceArray,
	GridInfo scGridInfo,
	NumVerticesGrid numVerticesGrid,
	ScalarFieldGrid ScalarFieldGrid,
	Float3Grid posGrid,
	Float3Grid norGrid,
	SimParam params)
{
	generateTriangles << <gridDim_, blockDim_ >> > (
		surfaceIndexInGridArray,
		validIndexInSurfaceArray,
		scGridInfo,
		numVerticesGrid,
		ScalarFieldGrid, 
		posGrid,
		norGrid,
		params);
	cudaDeviceSynchronize();
}

extern "C"
uint launchThrustExclusivePrefixSumScan(uint* output, uint* input, uint numElements)
{
	//! calculation of exclusive prefix sum.
	thrust::exclusive_scan(
		thrust::device_ptr<uint>(input),
		thrust::device_ptr<uint>(input + numElements),
		thrust::device_ptr<uint>(output));
	cudaDeviceSynchronize();

	uint lastElement = 0;
	uint lastElementScan = 0;
	checkCudaErrors(cudaMemcpy((void *)&lastElement, (void *)(input + numElements - 1),
		sizeof(uint), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((void *)&lastElementScan, (void *)(output + numElements - 1),
		sizeof(uint), cudaMemcpyDeviceToHost));
	uint sum = lastElement + lastElementScan;
	return sum;
}

void launchExtractionOfSurfaceAndInvolveParticles(
	dim3 gridDim_,
	dim3 blockDim_,
	SimParam simParam,
	DensityGrid flagGrid,
	NumInvolveParticlesGrid numInvolveParticlesGrid,
	IsSurfaceGrid surfaceParticlesFlagGrid,
	ParticleIndexRangeGrid particleIndexRangeGrid)
{
	extractionOfSurfaceAndInvolveParticles << < gridDim_, blockDim_ >> > (
		simParam,
		flagGrid,
		numInvolveParticlesGrid,
		surfaceParticlesFlagGrid,
		particleIndexRangeGrid);
	cudaDeviceSynchronize();
}

void launchEstimationOfSurfaceVertices(
	dim3 gridDim_,
	dim3 blockDim_,
	SimParam simParam,
	GridInfo scalarGridInfo,
	ParticleArray particleArray,
	IsSurfaceGrid surfaceParticlesFlagGrid,
	ParticleIndexRangeGrid particleIndexRangeGrid,
	IsSurfaceGrid isSurfaceGrid)
{
	estimationOfSurfaceVertices << < gridDim_, blockDim_ >> > (
		simParam,
		scalarGridInfo,
		particleArray,
		surfaceParticlesFlagGrid,
		particleIndexRangeGrid,
		isSurfaceGrid);
	cudaDeviceSynchronize();
}

void launchCalculationOfMeanAndSmoothParticles(
	dim3 gridDim_,
	dim3 blockDim_, 
	SimParam simParam,								
	ParticleArray particleArray,
	ParticleArray meanParticleArray,
	ParticleArray smoothedParticleArray,
	ParticleIndexRangeGrid particleIndexRangeGrid,
	GridInfo spatialGridInfo)
{
	calculationOfMeanAndSmoothParticles << < gridDim_, blockDim_ >> > (
		simParam,
		particleArray,
		meanParticleArray,
		smoothedParticleArray,
		particleIndexRangeGrid,
		spatialGridInfo);
	cudaDeviceSynchronize();
}

void launchCompactationOfSurfaceVerticesAndParticles(
	dim3 gridDim_,
	dim3 blockDim_,
	SimParam simParam,
	IsSurfaceGrid isSurfaceGrid,
	IsSurfaceGrid isSurfaceGridScan,
	NumInvolveParticlesGrid numInvolveParticlesGrid,
	NumInvolveParticlesGrid numInvolveParticlesGridScan,
	ParticleIndexRangeGrid particleIndexRangeGrid,
	SurfaceVerticesIndexArray surfaceVerticesIndexArray,
	SurfaceParticlesIndexArray surfaceParticlesIndexArray)
{
	compactationOfSurfaceVerticesAndParticles << < gridDim_, blockDim_ >> > (
		simParam,
		isSurfaceGrid,
		isSurfaceGridScan,
		numInvolveParticlesGrid,
		numInvolveParticlesGridScan,
		particleIndexRangeGrid,
		surfaceVerticesIndexArray,
		surfaceParticlesIndexArray);
	cudaDeviceSynchronize();
}

void launchCalculationOfTransformMatricesForParticles(
	dim3 gridDim_,
	dim3 blockDim_,
	SimParam simParam,
	GridInfo spatialGridInfo,
	ParticleArray meanParticleArray,
	ParticleArray particleArray,
	ParticleIndexRangeGrid particleIndexRangeGrid,
	SurfaceParticlesIndexArray surfaceParticlesIndexArray,
	MatrixArray svdMatricesArray)
{
	calculationOfTransformMatricesForParticles << < gridDim_, blockDim_ >> > (
		simParam,
		spatialGridInfo,
		meanParticleArray, 
		particleArray,
		particleIndexRangeGrid, 
		surfaceParticlesIndexArray,
		svdMatricesArray);
	cudaDeviceSynchronize();
}

void launchComputationOfScalarFieldGrid(
	dim3 gridDim_,
	dim3 blockDim_,
	SimParam simParam,
	uint numSurfaceVertices,
	GridInfo scalarGridInfo,
	GridInfo spatialGridInfo,
	MatrixArray svdMatricesArray,
	ParticleArray smoothedParticleArray,
	ScalarFieldGrid particlesDensityArray,
	ParticleIndexRangeGrid particleIndexRangeGrid,
	SurfaceVerticesIndexArray surfaceVerticesIndexArray,
	NumInvolveParticlesGrid numInvolveParticlesGrid,
	NumInvolveParticlesGrid numInvolveParticlesGridScan,
	ScalarFieldGrid scalarFieldGrid)
{
	computationOfScalarFieldGrid << < gridDim_, blockDim_ >> > (
		simParam,
		numSurfaceVertices,
		scalarGridInfo,
		spatialGridInfo,
		svdMatricesArray,
		smoothedParticleArray,
		particlesDensityArray,
		particleIndexRangeGrid,
		surfaceVerticesIndexArray,
		numInvolveParticlesGrid,
		numInvolveParticlesGridScan,
		scalarFieldGrid);
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
	ParticleIndexRangeGrid particlesIndexRangerArray,	// output, each cells' start index and end index.
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
			particlesIndexRangerArray.grid[hashValue].start = index;
			if (index > 0)
				particlesIndexRangerArray.grid[sharedHash[threadIdx.x]].end = index;
		}

		if (index == numParticles - 1)
			particlesIndexRangerArray.grid[hashValue].end = index + 1;
	}
}

void launchSpatialGridBuilding(
	ParticleArray particlesArray,
	ScalarFieldGrid densitiesArray,
	uint numParticles,
	ParticleIndexRangeGrid particlesIndexRangerArray,
	DensityGrid flagGrid,
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
		dGridParticleHash, particlesArray, numParticles, flagGrid, spatialGridInfo);
	getLastCudaError("calcParticlesHashKernel");
	cudaDeviceSynchronize();

	//! step2: sort the particle according to their hash value.
	thrust::device_ptr<ParticlePosition> posPtr(particlesArray.grid);
	thrust::device_ptr<ScalarValue> denPtr(densitiesArray.grid);
	thrust::sort_by_key(
		thrust::device_ptr<unsigned int>(dGridParticleHash),
		thrust::device_ptr<unsigned int>(dGridParticleHash + numParticles),
		thrust::make_zip_iterator(thrust::make_tuple(posPtr, denPtr)));
	getLastCudaError("sort_by_key");
	cudaDeviceSynchronize();

	//! step3: find start index and end index of each cell.
	// 0xffffffff, need to be attentioned.
	unsigned int memSize = sizeof(unsigned int) * (numThreads + 1);
	cudaMemset(particlesIndexRangerArray.grid, 0xffffffff, particlesIndexRangerArray.size * sizeof(IndexRange));
	findCellRangeKernel << < numBlocks, numThreads, memSize >> > (particlesIndexRangerArray, numParticles,
		dGridParticleHash);
	getLastCudaError("findCellRangeKernel");
	cudaDeviceSynchronize();

	//! freee memory.
	cudaFree(dGridParticleHash);
}
