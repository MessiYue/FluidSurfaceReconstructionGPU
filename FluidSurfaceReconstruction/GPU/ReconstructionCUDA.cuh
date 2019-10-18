#ifndef SURFACE_RECONSTRUCTION_CUH
#define SURFACE_RECONSTRUCTION_CUH

#include <helper_math.h>
#include <vector_types.h>

#include "Defines.h"
#include "CudaUtils.h"

const float EPSILON_CPU = 1.0e-7;
extern "C" __constant__ float SIGMA;
extern "C" __constant__ float EPSILON_;

//! ----------------------------------------launching functions--------------------------------------

//! func: bind auxiliary textures for marching cubes.
extern "C"
void bindTextures(
	uint* d_edgeTable,									// table of cube edges.
	int* d_edgeIndicesOfTriangleTable,					// table of cube edges' indices.
	uint* d_numVerticesTable,							// table of triangle vertices.
	uint* d_vertexIndicesOfEdgeTable);					// table of edge vertices' indices.

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
	ParticleIndexRangeGrid particleIndexRangeGrid,		// paritcle index information grid.
	ParticleArray particleArray,						// particles array.
	ScalarFieldGrid ScalarFieldGrid,								// vertex grid.
	GridInfo spatialGridInfo,							// spatial hashing grid information.
	GridInfo scalarGridInfo,							// scalar field grid information
	SimParam params);									// simulation parameters.

//! func: compact indices of surface vertices into 1D array.
extern "C" 
void launchCompactSurfaceVertex(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SurfaceVerticesIndexArray svIndexArray,				// compacted surface vertices' indices of grid.				
	IsSurfaceGrid isSurfaceGridScan,					// 
	IsSurfaceGrid isSurfaceGrid,						//
	SimParam params);									// simulation parameters.

//! func: update of compacted scalar gird.
extern "C" 
void launchUpdateScalarGridValuesCompacted(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SurfaceVerticesIndexArray svIndexArray,				// compacted surface vertices' indices of grid.
	uint numSurfaceVertices,							// number of surface vertices of grid.
	ParticleIndexRangeGrid particleIndexRangeGrid,		// paritcle index information grid.
	ParticleArray particleArray,						// particles array.
	ScalarFieldGrid ScalarFieldGrid,					// vertices of grid.
	GridInfo spatialGridInfo,							// spatial hashing grid information.
	GridInfo scalarGridInfo,							// scalar field grid information.
	SimParam params);							

//! func: detect valid surface cubes.
extern "C" 
void launchDetectValidSurfaceCubes(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SurfaceVerticesIndexArray svIndexArray,				// compacted surface vertices' indices of grid.
	uint numSurfaceVertices,							// number of surface vertices.
	ScalarFieldGrid vGrid,									// scalar field.
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
	SurfaceVerticesIndexArray surfaceIndexInGridArray,	// indices of surface vertices.
	ValidSurfaceIndexArray validIndexInSurfaceArray,	// compacted valid surface cubes.
	GridInfo scGridInfo,								// 
	NumVerticesGrid numVerticesGridScan,				// exculsize prefix sum of numVerticesGrid.
	ScalarFieldGrid ScalarFieldGrid,					// scalar field.
	Float3Grid posGrid,									// positions of obj
	Float3Grid norGrid,									// normals of obj
	SimParam params);

extern "C" uint 
launchThrustExclusivePrefixSumScan(uint* output, uint* input, uint numElements);

//! ------------------------------------ functions for anisotropic kernel------------------------------

//! func: calculation of mean and smoothed particles.
extern "C" void
launchCalculationOfMeanAndSmoothParticles(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SimParam simParam,									// simulation parameter.
	ParticleArray particleArray,						// particles array.
	ParticleArray meanParticleArray,					// mean postions of particles.
	ParticleArray smoothedParticleArray,				// smoothed positions of particles.
	ParticleIndexRangeGrid particleIndexRangeGrid,      // paritcle index information grid.
	GridInfo spatialGridInfo);

//! func: estimation of surface vertices and particles.
extern "C" void
launchEstimationOfSurfaceVerticesAndParticles(
	dim3 gridDim_,										// cuda grid dimension.
	dim3 blockDim_,										// cuda block dimension.
	SimParam simParam,									// simulation parameter.
	DensityGrid flagGrid,								// virtual density grid of particles.
	IsSurfaceGrid isSurfaceGrid,						// surface tag field.
	NumSurfaceParticlesGrid numSurfaceParticlesGrid,	// number of surface particles.
	ParticleIndexRangeGrid particleIndexRangeGrid,		// paritcle index information grid.
	GridInfo spatialGridInfo);

//! func: compactation of surface vertices and particles.
extern "C" void
launchCompactationOfSurfaceVerticesAndParticles(
	dim3 gridDim_,											// cuda grid dimension.
	dim3 blockDim_,											// cuda block dimension.
	SimParam simParam,										// simulation parameter.
	IsSurfaceGrid isSurfaceGrid,							// surface tag field.
	IsSurfaceGrid isSurfaceGridScan,						// exclusive prefix sum of isSurfaceGrid
	NumSurfaceParticlesGrid numSurfaceParticlesGrid,		// number of surface particles.
	NumSurfaceParticlesGrid numSurfaceParticlesGridScan,	// exclusive prefix sum of numSurfaceParticlesGrid.
	ParticleIndexRangeGrid particleIndexRangeGrid,			// paritcle index information grid.
	SurfaceVerticesIndexArray surfaceVerticesIndexArray,
	SurfaceParticlesIndexArray surfaceParticlesIndexArray);

//! func: calculation of transform matrices for each surface particle.
extern "C" void
launchCalculationOfTransformMatricesForParticles(
	dim3 gridDim_,											// cuda grid dimension.
	dim3 blockDim_,											// cuda block dimension.
	SimParam simParam,										// simulation parameter.
	GridInfo spatialGridInfo,								// spatial hashing grid information.
	ParticleArray meanParticleArray,						// mean postions of particles.
	ParticleArray particleArray,							// particles array.
	ParticleIndexRangeGrid particleIndexRangeGrid,			// paritcle index information grid.
	SurfaceParticlesIndexArray surfaceParticlesIndexArray,	// surface particles' indices.
	MatrixArray svdMatricesArray);							// transform matrices for each surface particle.

//! func: computation of scalar field grid using anisotropic kernel.
extern "C" void
launchComputationOfScalarFieldGrid(
	dim3 gridDim_,											// cuda grid dimension.
	dim3 blockDim_,											// cuda block dimension.
	SimParam simParam,										// simulation parameter.
	uint numSurfaceVertices,								// number of surface vertices.
	GridInfo scalarGridInfo,								// scalar field grid information.
	GridInfo spatialGridInfo,								// spatial hashing grid information.
	MatrixArray svdMatricesArray,							// transform matrices for each surface particle.
	ParticleArray smoothedParticleArray,					// smoothed postions of particles.
	ScalarFieldGrid particlesDensityArray,					// particle densities array.
	ParticleIndexRangeGrid particleIndexRangeGrid,			// paritcle index information grid.
	SurfaceVerticesIndexArray surfaceVerticesIndexArray,	// surface vertices' indices.
	NumSurfaceParticlesGrid numSurfaceParticlesGrid,		// number of surface particles.
	NumSurfaceParticlesGrid numSurfaceParticlesGridScan,	// exclusive prefix sum of numSurfaceParticlesGrid.
	ScalarFieldGrid ScalarFieldGrid);						// scalar field grid.


//! -----------------------------------Spatial Grid establish------------------------------------------
//! func: establish of spatial grid for neighborhood researching.
extern "C" void 
launchSpatialGridBuilding(
	ParticleArray particlesArray,						// particles' positions array.
	ScalarFieldGrid densitiesArray,						// particles' densities array.
	uint numParticles,									// number of particles.
	ParticleIndexRangeGrid particlesIndexRangerArray,	// particles' indices for each cell.
	DensityGrid densityGrid,							// virtual density grid.
	GridInfo spatialGridInfo);							// spatial hashing grid information.

//! -----------------------------------smoothing kernels------------------------------------------------

inline __host__ __device__
float kernelTC01(float distSq, float EffectiveRadiusSq)
{
	if (distSq > EffectiveRadiusSq)
		return 0.f;
	float d2DR2 = distSq / EffectiveRadiusSq * 0.5f;
	float d2DR2_2 = d2DR2 * d2DR2;
	return d2DR2_2 - d2DR2 + 0.25f;
}

inline __device__
float kernelZB05(float distSq, float effetiveRadiusSq)
{
	if (distSq >= effetiveRadiusSq)
		return 0;
	return pow(1.0f - distSq / effetiveRadiusSq, 3);
}

inline __host__ __device__
float weightFunc2(float distSq, float EffectiveRadiusSq)
{
	if (distSq > EffectiveRadiusSq)
		return 0.f;
	float t2 = distSq / EffectiveRadiusSq * 0.5f;
	float t4 = t2 * t2;
	float t8 = t4 * t4;
	return t8 - t4 + 0.1875f;
}

//! func: calculation of wij for equation (11) from Yu's paper.
inline __host__ __device__
float wij(float distance, float r)
{
	if (distance < r)
	{
		float s = distance / r;
		return 1.0f - s * s * s;
	}
	else
		return 0.0f;
}

//! func: anisotropic kernel function.
inline __host__ __device__
float anisotropicW(float3 r, MatrixValue g, float det)
{
	//! g * r.
	float3 target;
	target.x = r.x * g.a11 + r.y * g.a12 + r.z * g.a13;
	target.y = r.x * g.a21 + r.y * g.a22 + r.z * g.a23;
	target.z = r.x * g.a31 + r.y * g.a32 + r.z * g.a33;

	float dist = length(target);
	float distSq = dist * dist;
	if (distSq >= 1.0)
		return 0.0f;
	float x = 1.0 - distSq;
	return SIGMA * det * x * x * x;
}

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
int3 getIndex3D(float3 pos, float3 gridMinPos, float cellSize)
{
	return make_int3((pos - gridMinPos) / cellSize);
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
float getValue(int3 index3D, ScalarFieldGrid vGrid)
{
	if (isValid(index3D, vGrid.resolution))
		return vGrid.grid[index3DTo1D(make_uint3(index3D), vGrid.resolution)].value;
	return 0.f;
}

//inline __host__ __device__ 
//float getValue(int3 index3D, IntGrid sfVerIndexGrid, VerAkGrid sfVerGrid)
//{
//	if (isValid(index3D, sfVerIndexGrid.resolution))
//	{
//		uint index1D = index3DTo1D(make_uint3(index3D), sfVerIndexGrid.resolution);
//		int indexSfGrid = sfVerIndexGrid.grid[index1D];
//		if (indexSfGrid >= 0 && indexSfGrid < sfVerGrid.count)
//			return sfVerGrid.grid[indexSfGrid].value;
//		return 0.f;
//	}
//
//	return 0.f;
//}

inline __host__ __device__ 
float3 getVertexNorm(uint3 index3D, ScalarFieldGrid vGrid)
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

//inline __host__ __device__ 
//float3 getVertexNorm(uint3 index3D, IntGrid sfVerIndexGrid, VerAkGrid sfVerGrid)
//{
//	int i = index3D.x;
//	int j = index3D.y;
//	int k = index3D.z;
//	float3 n;
//	n.x = getValue(make_int3(i - 1, j, k), sfVerIndexGrid, sfVerGrid) -
//		getValue(make_int3(i + 1, j, k), sfVerIndexGrid, sfVerGrid);
//
//	n.y = getValue(make_int3(i, j - 1, k), sfVerIndexGrid, sfVerGrid) -
//		getValue(make_int3(i, j + 1, k), sfVerIndexGrid, sfVerGrid);
//
//	n.z = getValue(make_int3(i, j, k - 1), sfVerIndexGrid, sfVerGrid) -
//		getValue(make_int3(i, j, k + 1), sfVerIndexGrid, sfVerGrid);
//
//	n = normalize(n);
//	return n;
//}

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
uint getVertexFlag(uint3 curIndex3D, ScalarFieldGrid ScalarFieldGrid, float isoValue)
{
	// for marching cube algorithm.
	uint cornerIndex[8];
	cornerIndex[0] = index3DTo1D(curIndex3D + make_uint3(0, 0, 0), ScalarFieldGrid.resolution);
	cornerIndex[1] = index3DTo1D(curIndex3D + make_uint3(0, 0, 1), ScalarFieldGrid.resolution);
	cornerIndex[2] = index3DTo1D(curIndex3D + make_uint3(0, 1, 1), ScalarFieldGrid.resolution);
	cornerIndex[3] = index3DTo1D(curIndex3D + make_uint3(0, 1, 0), ScalarFieldGrid.resolution);
	cornerIndex[4] = index3DTo1D(curIndex3D + make_uint3(1, 0, 0), ScalarFieldGrid.resolution);
	cornerIndex[5] = index3DTo1D(curIndex3D + make_uint3(1, 0, 1), ScalarFieldGrid.resolution);
	cornerIndex[6] = index3DTo1D(curIndex3D + make_uint3(1, 1, 1), ScalarFieldGrid.resolution);
	cornerIndex[7] = index3DTo1D(curIndex3D + make_uint3(1, 1, 0), ScalarFieldGrid.resolution);

	uint vertexFlag = 0;//低8位用于标志8个顶点与等值面的位置关系

	for (size_t i = 0; i < 8; i++)
		if (ScalarFieldGrid.grid[cornerIndex[i]].value <= isoValue)//位于等值面上或等值面的一侧
			vertexFlag |= 1 << i;
	return vertexFlag;
}

inline __host__ __device__ 
uint getVertexFlag(uint cornerIndex[8], ScalarFieldGrid ScalarFieldGrid, float isoValue)
{
	// for marching cube algorithm.
	uint vertexFlag = 0;//低8位用于标志8个顶点与等值面的位置关系

	for (size_t i = 0; i < 8; i++)
		if (ScalarFieldGrid.grid[cornerIndex[i]].value <= isoValue)//位于等值面上或等值面的一侧
			vertexFlag |= 1 << i;
	return vertexFlag;
}

//inline __host__ __device__ 
//uint getVertexFlag(uint* cornerIndexScGrid, IntGrid sfVerIndexGrid, VerAkGrid sfVerGrid, float isoValue)
//{
//	uint vertexFlag = 0;//低8位用于标志8个顶点与等值面的位置关系
//
//	for (size_t i = 0; i < 8; i++)
//	{
//		float value = 0.f;//非表面顶点值默认为0
//		int indexSf = sfVerIndexGrid.grid[cornerIndexScGrid[i]];
//		if (indexSf >= 0)//表面顶点
//			value = sfVerGrid.grid[indexSf].value;
//		if (value <= isoValue)//位于等值面上或等值面的一侧
//			vertexFlag |= 1 << i;
//	}
//	return vertexFlag;
//}

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
void getCornerNormals(uint3* cornerIndex3Ds, ScalarFieldGrid vGrid, float3* cornerNormals)
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

//inline __host__ __device__ 
//void getCornerNormals(uint3* cornerIndex3Ds, IntGrid sfVerIndexGrid, VerAkGrid sfVerGrid, float3* cornerNormals)
//{
//	cornerNormals[0] = getVertexNorm(cornerIndex3Ds[0], sfVerIndexGrid, sfVerGrid);
//	cornerNormals[1] = getVertexNorm(cornerIndex3Ds[1], sfVerIndexGrid, sfVerGrid);
//	cornerNormals[2] = getVertexNorm(cornerIndex3Ds[2], sfVerIndexGrid, sfVerGrid);
//	cornerNormals[3] = getVertexNorm(cornerIndex3Ds[3], sfVerIndexGrid, sfVerGrid);
//	cornerNormals[4] = getVertexNorm(cornerIndex3Ds[4], sfVerIndexGrid, sfVerGrid);
//	cornerNormals[5] = getVertexNorm(cornerIndex3Ds[5], sfVerIndexGrid, sfVerGrid);
//	cornerNormals[6] = getVertexNorm(cornerIndex3Ds[6], sfVerIndexGrid, sfVerGrid);
//	cornerNormals[7] = getVertexNorm(cornerIndex3Ds[7], sfVerIndexGrid, sfVerGrid);
//}

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

__device__ __forceinline__
void matrixMul(
	float a11, float a12, float a13,
	float a21, float a22, float a23,
	float a31, float a32, float a33,
	float b11, float b12, float b13,
	float b21, float b22, float b23,
	float b31, float b32, float b33,
	float &c11, float &c12, float &c13,
	float &c21, float &c22, float &c23,
	float &c31, float &c32, float &c33)
{
	c11 = a11 * b11 + a12 * b21 + a13 * b31;
	c21 = a21 * b11 + a22 * b21 + a23 * b31;
	c31 = a31 * b11 + a32 * b21 + a33 * b31;

	c12 = a11 * b12 + a12 * b22 + a13 * b32;
	c22 = a21 * b12 + a22 * b22 + a23 * b32;
	c32 = a31 * b12 + a32 * b22 + a33 * b32;

	c13 = a11 * b13 + a12 * b23 + a13 * b33;
	c23 = a21 * b13 + a22 * b23 + a23 * b33;
	c33 = a31 * b13 + a32 * b23 + a33 * b33;
}

//! func: calculation of determinant for 3x3 matrix.
__device__ __forceinline__
float determinant(const MatrixValue& mat)
{
	return
		mat.a11 * mat.a22 * mat.a33 + 
		mat.a12 * mat.a23 * mat.a31 + 
		mat.a13 * mat.a21 * mat.a32 - 
		mat.a11 * mat.a23 * mat.a32 -
		mat.a12 * mat.a21 * mat.a33 - 
		mat.a13 * mat.a22 * mat.a31;
}

#endif