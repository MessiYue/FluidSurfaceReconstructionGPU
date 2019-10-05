#ifndef SURFACE_RECONSTRUCTION_CUH
#define SURFACE_RECONSTRUCTION_CUH

#include <helper_math.h>
#include <vector_types.h>

#include "Defines.h"
#include "CudaUtils.h"
#include "Kernel.cuh"

//! epsilon pof float number. 
extern "C" __constant__ float EPSILON_;
const float EPSILON_CPU = 1.0e-7;

//! func: estimation of surface vertices
extern "C"
void launchEstimateSurfaceVertices(
	dim3 gridDim,										// cuda grid dimension.
	dim3 blockDim,										// cuda block dimension.
	DensityGrid densityGrid,							// virtual density grid of particles.
	IsSurfaceGrid isSurfaceGrid,						// surface tag field.
	uint scSearchExt,									// surface cell?
	SimParam params);									// simulation parameters.

//! func: computation of scalar field.
extern "C" 
void launchUpdateScalarGridValues(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	IsSurfaceGrid isSurfaceGrid,						// surface tag field.
	ParticleIndexInfoGrid particleIndexInfoGrid,		// paritcle index information grid.
	ParticleArray particleArray,						// particles array.
	VertexGrid vertexGrid,								// vertex grid.
	GridInfo spatialGridInfo,							// spatial hashing grid information.
	GridInfo scalarGridInfo,							// scalar field grid information
	SimParam params);									// simulation parameters.

//! func: compact indices of surface vertices into 1D array.
extern "C" 
void launchCompactSurfaceVertex(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SurfaceVertexIndexArray svIndexArray,				// compacted surface vertices' indices of grid.				
	IsSurfaceGrid isSurfaceGridScan,					// 
	IsSurfaceGrid isSurfaceGrid,						//
	SimParam params);									// simulation parameters.

//! func: update of compacted scalar gird.
extern "C" 
void launchUpdateScalarGridValuesCompacted(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SurfaceVertexIndexArray svIndexArray,				// compacted surface vertices' indices of grid.
	uint numSurfaceVertices,							// number of surface vertices of grid.
	ParticleIndexInfoGrid particleIndexInfoGrid,		// paritcle index information grid.
	ParticleArray particleArray,						// particles array.
	VertexGrid vertexGrid,								// vertices of grid.
	GridInfo spatialGridInfo,							// spatial hashing grid information.
	GridInfo scalarGridInfo,							// scalar field grid information.
	SimParam params);							

//! func: 
extern "C" 
void bindTextures(
	uint* d_edgeTable,									// table of cube edges.
	int* d_edgeIndicesOfTriangleTable,					// table of cube edges' indices.
	uint* d_numVerticesTable,							// table of triangle vertices.
	uint* d_vertexIndicesOfEdgeTable);					// table of edge vertices' indices.

//! func: detect valid surface cubes.
extern "C" 
void launchDetectValidSurfaceCubes(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SurfaceVertexIndexArray svIndexArray,				// compacted surface vertices' indices of grid.
	uint numSurfaceVertices,							// number of surface vertices.
	VertexGrid vGrid,									// scalar field.
	IsValidSurfaceGrid isValidSurfaceGrid,				// valid tag of surface cubes.
	NumVerticesGrid numVerticesGrid,					// number of vertices per cell of grid.
	IsSurfaceGrid isSfGrid,								// surface vertex tag.
	SimParam params);							

//! func: compact valid surface cubes.
extern "C" 
void launchCompactValidSurfaceCubes(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	ValidSurfaceIndexArray indexArray,					// valid cubes' indices.
	IsValidSurfaceGrid validSurfaceGridScan,			// exculsize prefix sum of validSurafceGrid.
	IsValidSurfaceGrid validSurafceGrid,				// cubes' valid tags
	SimParam params);				

//! func: generate triangles for fluid surface.
extern "C" void
launchGenerateTriangles(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SurfaceVertexIndexArray surfaceIndexInGridArray,	// indices of surface vertices.
	ValidSurfaceIndexArray validIndexInSurfaceArray,	// compacted valid surface cubes.
	GridInfo scGridInfo,								// 
	NumVerticesGrid numVerticesGridScan,				// exculsize prefix sum of numVerticesGrid.
	VertexGrid vertexGrid,								// scalar field.
	Float3Grid posGrid,									// positions of obj
	Float3Grid norGrid,									// normals of obj
	SimParam params);								

//! -----------------------------------Spatial Grid establish------------------------------------------
//! func: establish of spatial grid for neighborhood researching.
extern "C" void 
launchSpatialGridBuilding(
	ParticleArray *particlesArray,						// particles array.
	uint numParticles,									// number of particles.
	ParticleIndexInfoGrid *particlesIndexInforArray,	// particles' indices for each cell.
	DensityGrid *densityGrid,							// virtual density grid.
	GridInfo spatialGridInfo);							// spatial hashing grid information.


//! ------------------------------------Our method-----------------------------------------------------

//! Surface vertices estimation of MC grid. 
struct EstSfVer2016
{
	// Spatial hashing grid for particles' neighborhood search.
	IntGrid parStartInCellGrid;
	IntGrid parCountInCellGrid;
	UintGrid parIndexArray;

	//! particles' density.
	FloatGrid parDenArray;

	//! is surface vertex.
	UintGrid isSfGrid;

	//! ???.
	uint scSearchExt;
};

struct UpdateScGridV2016
{
	//! compacted surface vertices' indices.
	UintGrid sfVerIdxScArray;

	//! spatial hashing grid for particles' neighborhood search.
	IntGrid parStartInCellGrid;
	IntGrid parCountInCellGrid;
	UintGrid parIndexArray;
	Float3Grid parPosArray;

	// scalar field grid.
	VertexGrid verGrid;

	//! sptial hasing grid information.
	GridInfo spGridInfo;
	//! mc grid information.
	GridInfo scGridInfo;
	//! number of surface vertices.
	uint numSfVer;
};

extern "C" 
void launchEstimateSurfaceVertices2016(
	dim3 gridDim,
	dim3 blockDim,
	EstSfVer2016 param);

extern "C" 
void launchUpdateScalarGridValuesCompacted2016(
	dim3 gridDim_,
	dim3 blockDim_,
	UpdateScGridV2016 param);

extern "C" 
void launchUpdateScalarGridValuesCompacted2016ZB05(
	dim3 gridDim_,
	dim3 blockDim_, 
	UpdateScGridV2016 param);

//! ------------------------------------------Common---------------------------------------------

inline __host__ __device__ 
uint index3DTo1D(uint3 index3, uint3 res)
{
	return index3.z*res.x*res.y + index3.y*res.x + index3.x;
}

inline __host__ __device__ 
uint3 index1DTo3D(uint index1, uint3 res)
{
	uint z = index1 / (res.x*res.y);
	uint m = index1 % (res.x*res.y);
	uint y = m / res.x;
	uint x = m % res.x;
	return make_uint3(x, y, z);
}

inline __host__ __device__ 
uint3 getIndex3D(float3 pos, float3 gridMinPos, float cellSize)
{
	return make_uint3((pos - gridMinPos) / cellSize);
}

inline __host__ __device__ 
int3 getIndex3DSigned(float3 pos, float3 gridMinPos, float cellSize)
{
	return make_int3((pos - gridMinPos) / cellSize);
}

inline __host__ __device__ 
float3 getVertexPos(uint3 index3D, float3 gridMinPos, float cellSize)
{
	return make_float3(index3D)*cellSize + gridMinPos;
}

inline  __device__ 
float getLerpFac(float val0, float val1, float targetVal)
{
	float delta = val1 - val0;
	if (delta > -EPSILON_ && delta < EPSILON_)
		return 0.5f;
	return (targetVal - val0) / delta;
}

inline __host__  
float getLerpFacCPU(float val0, float val1, float targetVal)
{
	float delta = val1 - val0;
	if (delta > -EPSILON_CPU && delta < EPSILON_CPU)
		return 0.5f;
	return (targetVal - val0) / delta;
}

inline  __device__ 
float isZero1(float a)
{
	return (a > -EPSILON_ && a < EPSILON_);
}

inline  __device__ 
float isZero2(float a)
{
	return (a > -1e-16 && a < 1e-16);
}

inline __host__ __device__ 
uint isValid(int3 index3D, uint3 resolution)
{
	return (index3D.x >= 0 && (uint)index3D.x < resolution.x &&
		index3D.y >= 0 && (uint)index3D.y < resolution.y &&
		index3D.z >= 0 && (uint)index3D.z < resolution.z);
}

inline __host__ __device__ 
float getValue(int3 index3D, VertexGrid vGrid)
{
	if (isValid(index3D, vGrid.resolution))
		return vGrid.grid[index3DTo1D(make_uint3(index3D), vGrid.resolution)].value;
	return 0.f;
}

inline __host__ __device__ 
float getValue(int3 index3D, IntGrid sfVerIndexGrid, VerAkGrid sfVerGrid)
{
	if (isValid(index3D, sfVerIndexGrid.resolution))
	{
		uint index1D = index3DTo1D(make_uint3(index3D), sfVerIndexGrid.resolution);
		int indexSfGrid = sfVerIndexGrid.grid[index1D];
		if (indexSfGrid >= 0 && indexSfGrid < sfVerGrid.count)
			return sfVerGrid.grid[indexSfGrid].value;
		return 0.f;
	}

	return 0.f;
}

inline __host__ __device__ 
float3 getVertexNorm(uint3 index3D, VertexGrid vGrid)
{
	int i = index3D.x;
	int j = index3D.y;
	int k = index3D.z;
	float3 n;
	n.x = getValue(make_int3(i - 1, j, k), vGrid) - getValue(make_int3(i + 1, j, k), vGrid);
	n.y = getValue(make_int3(i, j - 1, k), vGrid) - getValue(make_int3(i, j + 1, k), vGrid);
	n.z = getValue(make_int3(i, j, k - 1), vGrid) - getValue(make_int3(i, j, k + 1), vGrid);
	n = normalize(n);
	return n;
}

inline __host__ __device__ 
float3 getVertexNorm(uint3 index3D, IntGrid sfVerIndexGrid, VerAkGrid sfVerGrid)
{
	int i = index3D.x;
	int j = index3D.y;
	int k = index3D.z;
	float3 n;
	n.x = getValue(make_int3(i - 1, j, k), sfVerIndexGrid, sfVerGrid) -
		getValue(make_int3(i + 1, j, k), sfVerIndexGrid, sfVerGrid);

	n.y = getValue(make_int3(i, j - 1, k), sfVerIndexGrid, sfVerGrid) -
		getValue(make_int3(i, j + 1, k), sfVerIndexGrid, sfVerGrid);

	n.z = getValue(make_int3(i, j, k - 1), sfVerIndexGrid, sfVerGrid) -
		getValue(make_int3(i, j, k + 1), sfVerIndexGrid, sfVerGrid);

	n = normalize(n);
	return n;
}

inline __host__ __device__
float3 calcNormal(float3 v0, float3 v1, float3 v2)
{
	float3 edge0 = v1 - v0;
	float3 edge1 = v2 - v0;
	// note - it's faster to perform normalization in vertex shader rather than here
	return cross(edge0, edge1);
}

inline __host__ __device__ 
void getCornerIndex3Ds(uint3 curIndex3D, uint3* cornerIndex3Ds)
{
	cornerIndex3Ds[0] = curIndex3D + make_uint3(0, 0, 0);
	cornerIndex3Ds[1] = curIndex3D + make_uint3(0, 0, 1);
	cornerIndex3Ds[2] = curIndex3D + make_uint3(0, 1, 1);
	cornerIndex3Ds[3] = curIndex3D + make_uint3(0, 1, 0);
	cornerIndex3Ds[4] = curIndex3D + make_uint3(1, 0, 0);
	cornerIndex3Ds[5] = curIndex3D + make_uint3(1, 0, 1);
	cornerIndex3Ds[6] = curIndex3D + make_uint3(1, 1, 1);
	cornerIndex3Ds[7] = curIndex3D + make_uint3(1, 1, 0);
}

inline __host__ __device__ 
uint getNeighborIndex3Ds(uint3 resolution, uint3 curIndex3D, uint3* neighIndex3Ds, uint maxCount)
{
	uint count = 0;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			for (int k = -1; k <= 1; k++)
			{
				int3 indexTry = make_int3(curIndex3D.x + i, curIndex3D.y + j, curIndex3D.z + k);
				if (isValid(indexTry, resolution))
				{
					if (count >= maxCount)
						return count;
					neighIndex3Ds[count++] = make_uint3(indexTry);
				}
			}
		}
	}
	return count;
}

inline __host__ __device__ 
void getCornerIndex1Ds(uint3 curIndex3D, uint3 resolution, uint* cornerIndex1Ds)
{
	cornerIndex1Ds[0] = index3DTo1D(curIndex3D + make_uint3(0, 0, 0), resolution);
	cornerIndex1Ds[1] = index3DTo1D(curIndex3D + make_uint3(0, 0, 1), resolution);
	cornerIndex1Ds[2] = index3DTo1D(curIndex3D + make_uint3(0, 1, 1), resolution);
	cornerIndex1Ds[3] = index3DTo1D(curIndex3D + make_uint3(0, 1, 0), resolution);
	cornerIndex1Ds[4] = index3DTo1D(curIndex3D + make_uint3(1, 0, 0), resolution);
	cornerIndex1Ds[5] = index3DTo1D(curIndex3D + make_uint3(1, 0, 1), resolution);
	cornerIndex1Ds[6] = index3DTo1D(curIndex3D + make_uint3(1, 1, 1), resolution);
	cornerIndex1Ds[7] = index3DTo1D(curIndex3D + make_uint3(1, 1, 0), resolution);
}

inline __host__ __device__ 
uint getVertexFlag(uint3 curIndex3D, FloatGrid valueGrid, float isoValue)
{
	// for marching cube algorithm.
	uint cornerIndex[8];
	cornerIndex[0] = index3DTo1D(curIndex3D + make_uint3(0, 0, 0), valueGrid.resolution);
	cornerIndex[1] = index3DTo1D(curIndex3D + make_uint3(0, 0, 1), valueGrid.resolution);
	cornerIndex[2] = index3DTo1D(curIndex3D + make_uint3(0, 1, 1), valueGrid.resolution);
	cornerIndex[3] = index3DTo1D(curIndex3D + make_uint3(0, 1, 0), valueGrid.resolution);
	cornerIndex[4] = index3DTo1D(curIndex3D + make_uint3(1, 0, 0), valueGrid.resolution);
	cornerIndex[5] = index3DTo1D(curIndex3D + make_uint3(1, 0, 1), valueGrid.resolution);
	cornerIndex[6] = index3DTo1D(curIndex3D + make_uint3(1, 1, 1), valueGrid.resolution);
	cornerIndex[7] = index3DTo1D(curIndex3D + make_uint3(1, 1, 0), valueGrid.resolution);

	int vertexFlag = 0;//低8位用于标志8个顶点与等值面的位置关系

	for (size_t i = 0; i < 8; i++)
		if (valueGrid.grid[cornerIndex[i]] <= isoValue)//位于等值面上或等值面的一侧
			vertexFlag |= 1 << i;
	return vertexFlag;
}

inline __host__ __device__ 
uint getVertexFlag(uint3 curIndex3D, VertexGrid vertexGrid, float isoValue)
{
	// for marching cube algorithm.
	uint cornerIndex[8];
	cornerIndex[0] = index3DTo1D(curIndex3D + make_uint3(0, 0, 0), vertexGrid.resolution);
	cornerIndex[1] = index3DTo1D(curIndex3D + make_uint3(0, 0, 1), vertexGrid.resolution);
	cornerIndex[2] = index3DTo1D(curIndex3D + make_uint3(0, 1, 1), vertexGrid.resolution);
	cornerIndex[3] = index3DTo1D(curIndex3D + make_uint3(0, 1, 0), vertexGrid.resolution);
	cornerIndex[4] = index3DTo1D(curIndex3D + make_uint3(1, 0, 0), vertexGrid.resolution);
	cornerIndex[5] = index3DTo1D(curIndex3D + make_uint3(1, 0, 1), vertexGrid.resolution);
	cornerIndex[6] = index3DTo1D(curIndex3D + make_uint3(1, 1, 1), vertexGrid.resolution);
	cornerIndex[7] = index3DTo1D(curIndex3D + make_uint3(1, 1, 0), vertexGrid.resolution);

	uint vertexFlag = 0;//低8位用于标志8个顶点与等值面的位置关系

	for (size_t i = 0; i < 8; i++)
		if (vertexGrid.grid[cornerIndex[i]].value <= isoValue)//位于等值面上或等值面的一侧
			vertexFlag |= 1 << i;
	return vertexFlag;
}

inline __host__ __device__ 
uint getVertexFlag(uint cornerIndex[8], VertexGrid vertexGrid, float isoValue)
{
	// for marching cube algorithm.
	uint vertexFlag = 0;//低8位用于标志8个顶点与等值面的位置关系

	for (size_t i = 0; i < 8; i++)
		if (vertexGrid.grid[cornerIndex[i]].value <= isoValue)//位于等值面上或等值面的一侧
			vertexFlag |= 1 << i;
	return vertexFlag;
}


inline __host__ __device__ 
uint getVertexFlag(uint* cornerIndexScGrid, IntGrid sfVerIndexGrid, VerAkGrid sfVerGrid, float isoValue)
{
	uint vertexFlag = 0;//低8位用于标志8个顶点与等值面的位置关系

	for (size_t i = 0; i < 8; i++)
	{
		float value = 0.f;//非表面顶点值默认为0
		int indexSf = sfVerIndexGrid.grid[cornerIndexScGrid[i]];
		if (indexSf >= 0)//表面顶点
			value = sfVerGrid.grid[indexSf].value;
		if (value <= isoValue)//位于等值面上或等值面的一侧
			vertexFlag |= 1 << i;
	}
	return vertexFlag;
}


inline __host__ __device__ 
void getCornerPositions(uint3* cornerIndex3Ds, float3 gridMinPos, float cellSize, float3* cornerPoss)
{
	cornerPoss[0] = getVertexPos(cornerIndex3Ds[0], gridMinPos, cellSize);
	cornerPoss[1] = getVertexPos(cornerIndex3Ds[1], gridMinPos, cellSize);
	cornerPoss[2] = getVertexPos(cornerIndex3Ds[2], gridMinPos, cellSize);
	cornerPoss[3] = getVertexPos(cornerIndex3Ds[3], gridMinPos, cellSize);
	cornerPoss[4] = getVertexPos(cornerIndex3Ds[4], gridMinPos, cellSize);
	cornerPoss[5] = getVertexPos(cornerIndex3Ds[5], gridMinPos, cellSize);
	cornerPoss[6] = getVertexPos(cornerIndex3Ds[6], gridMinPos, cellSize);
	cornerPoss[7] = getVertexPos(cornerIndex3Ds[7], gridMinPos, cellSize);
}

inline __host__ __device__ 
void getCornerNormals(uint3* cornerIndex3Ds, VertexGrid vGrid, float3* cornerNormals)
{
	cornerNormals[0] = getVertexNorm(cornerIndex3Ds[0], vGrid);
	cornerNormals[1] = getVertexNorm(cornerIndex3Ds[1], vGrid);
	cornerNormals[2] = getVertexNorm(cornerIndex3Ds[2], vGrid);
	cornerNormals[3] = getVertexNorm(cornerIndex3Ds[3], vGrid);
	cornerNormals[4] = getVertexNorm(cornerIndex3Ds[4], vGrid);
	cornerNormals[5] = getVertexNorm(cornerIndex3Ds[5], vGrid);
	cornerNormals[6] = getVertexNorm(cornerIndex3Ds[6], vGrid);
	cornerNormals[7] = getVertexNorm(cornerIndex3Ds[7], vGrid);
}

inline __host__ __device__ 
void getCornerNormals(uint3* cornerIndex3Ds, IntGrid sfVerIndexGrid, VerAkGrid sfVerGrid, float3* cornerNormals)
{
	cornerNormals[0] = getVertexNorm(cornerIndex3Ds[0], sfVerIndexGrid, sfVerGrid);
	cornerNormals[1] = getVertexNorm(cornerIndex3Ds[1], sfVerIndexGrid, sfVerGrid);
	cornerNormals[2] = getVertexNorm(cornerIndex3Ds[2], sfVerIndexGrid, sfVerGrid);
	cornerNormals[3] = getVertexNorm(cornerIndex3Ds[3], sfVerIndexGrid, sfVerGrid);
	cornerNormals[4] = getVertexNorm(cornerIndex3Ds[4], sfVerIndexGrid, sfVerGrid);
	cornerNormals[5] = getVertexNorm(cornerIndex3Ds[5], sfVerIndexGrid, sfVerGrid);
	cornerNormals[6] = getVertexNorm(cornerIndex3Ds[6], sfVerIndexGrid, sfVerGrid);
	cornerNormals[7] = getVertexNorm(cornerIndex3Ds[7], sfVerIndexGrid, sfVerGrid);
}

inline __host__ __device__ 
uint isAllSfVertex(uint* corIndex1Ds, IsSurfaceGrid isSfGrid)
{
	for (int i = 0; i < 8; i++)
		if (!isSfGrid.grid[corIndex1Ds[i]])
			return FALSE_;
	return TRUE_;
}

inline __host__ __device__ 
uint isAllSfVertex(uint* corIndex1Ds, IntGrid sfIndexGrid)
{
	for (int i = 0; i < 8; i++)
	{
		if (sfIndexGrid.grid[corIndex1Ds[i]] < 0)
			return FALSE_;
	}
	return TRUE_;
}

//new version
inline __host__ __device__ 
uint getVertexFlag(uint3 curIndex3D, IntGrid parStartInCellGrid, IntGrid parCountInCellGrid,
	UintGrid parIndexArray, FloatGrid parDenArray, float isoValue)
{
	uint cornerIndex[8];
	//corner cell index
	cornerIndex[0] = index3DTo1D(curIndex3D + make_uint3(0, 0, 0), parStartInCellGrid.resolution);
	cornerIndex[1] = index3DTo1D(curIndex3D + make_uint3(0, 0, 1), parStartInCellGrid.resolution);
	cornerIndex[2] = index3DTo1D(curIndex3D + make_uint3(0, 1, 1), parStartInCellGrid.resolution);
	cornerIndex[3] = index3DTo1D(curIndex3D + make_uint3(0, 1, 0), parStartInCellGrid.resolution);
	cornerIndex[4] = index3DTo1D(curIndex3D + make_uint3(1, 0, 0), parStartInCellGrid.resolution);
	cornerIndex[5] = index3DTo1D(curIndex3D + make_uint3(1, 0, 1), parStartInCellGrid.resolution);
	cornerIndex[6] = index3DTo1D(curIndex3D + make_uint3(1, 1, 1), parStartInCellGrid.resolution);
	cornerIndex[7] = index3DTo1D(curIndex3D + make_uint3(1, 1, 0), parStartInCellGrid.resolution);

	uint vertexFlag = 0;//低8位用于标志8个顶点与等值面的位置关系
	for (int i = 0; i < 8; i++)
	{
		float den = 0.f;
		if (parCountInCellGrid.grid[cornerIndex[i]] > 0)
		{
			//printf("corcell=%d ", cornerIndex[i]);

			//cell start particle index in parIndexArray
			cornerIndex[i] = parStartInCellGrid.grid[cornerIndex[i]];

			//printf("start=%d ", cornerIndex[i]);

			//cell start particle index in particle buffer
			cornerIndex[i] = parIndexArray.grid[cornerIndex[i]];

			//printf("parIdx=%d ", cornerIndex[i]);
			den = parDenArray.grid[cornerIndex[i]];
		}
		if (den <= isoValue)//位于等值面上或等值面的一侧
			vertexFlag |= 1 << i;
	}

	return vertexFlag;
}


#endif