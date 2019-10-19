#pragma once

#include <cstdlib>
#include <iostream>
#include <vector_types.h>
#include <vector_functions.h>

#include "MathVector.h"

#define TRUE_ 1
#define FALSE_ 0

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long long ulonglong;

//! simulation parameters.
struct SimParam
{
	float particleRadius;				// particle radius.
	float particleMass;					// particle mass.
	float smoothingRadius;				// effective radius.
	float smoothingRadiusSq;			// square of effective radius.
	float smoothingRadiusInv;			// 1.0 / smoothingRadius.
	float lambdaForSmoothed;			//
	float anisotropicRadius;			// r = 2 * smoothingRadius.
	float isoValue;						// isocontour value
	uint scSpGridResRatio;				// spatial hashing grid res / scalar field grid res.
	uint expandExtent;					// expandsion extent.
	uint minNumNeighbors;				// minimal number of neighbors for non isolated particle.
	float spatialCellSizeScale;			// scaling for spatial hashing grid cell size.
};

//! grid or array type.
template<typename T>
struct Grid 
{
	T* grid;													// data array.
	ulonglong size;												// array size.
	ulonglong count;											//
	uint3 resolution;											// grid resolution.
};

//! index range from start to end.
struct IndexRange { uint start, end; };

//! particle position.
struct ParticlePosition { float3 pos; };

//! scalar value.
struct ScalarValue { float value; };

//! 3x3 matrix.
struct MatrixValue 
{
	float a11, a12, a13;
	float a21, a22, a23;
	float a31, a32, a33;
};

//! grid's size information.
struct GridInfo
{
	float3 minPos;
	float3 maxPos;
	float cellSize;
	uint3 resolution;
};

//! triangle.
struct Triangle
{
	fVector3 vertices[3];
	fVector3 normals[3];
};

typedef Grid<int> IntGrid;
typedef Grid<uint> UintGrid;
typedef Grid<float> FloatGrid;
typedef Grid<float3> Float3Grid;
typedef Grid<float> DensityGrid;
typedef Grid<ParticlePosition> ParticleArray;
typedef Grid<IndexRange> ParticleIndexRangeGrid;
typedef Grid<ScalarValue>  ScalarFieldGrid;			
typedef Grid<uint> IsSurfaceGrid;
typedef Grid<uint> NumInvolveParticlesGrid;
typedef Grid<uint> NumVerticesGrid;								// number of vertices.
typedef Grid<uint> IsValidSurfaceGrid;							// whethe valid or not for cells.
typedef Grid<uint> SurfaceParticlesIndexArray;
typedef Grid<uint> SurfaceVerticesIndexArray;					// compacted surface vertices' indices.
typedef Grid<uint> ValidSurfaceIndexArray;						// compacted valid surface cells' indices.
typedef Grid<MatrixValue> MatrixArray;							

inline void safeFree(void** ptr)
{
	if (ptr == 0 || *ptr == 0)
		return;
	free(*ptr);
	*ptr = 0;
}

#ifndef SAFE_DELETE
#define SAFE_DELETE(p) {if(p){delete p;p=0;}}
#endif

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) {if(p){delete []p;p=0;}}
#endif 

#define INITGRID_ZERO(GRID)	GRID.grid=0;\
	GRID.resolution = make_uint3(0, 0, 0); \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = 0;

#define CUDA_CREATE_GRID_1D(GRID,SIZE_,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = make_uint3(SIZE_, 1, 1); \
	GRID.size = SIZE_; \
	GRID.count = 0; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE)));

#define CUDA_CREATE_GRID_3D_S(GRID,SIZEX,SIZEY,SIZEZ,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = make_uint3(SIZEX, SIZEY, SIZEZ); \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = 0; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE)));

#define CUDA_CREATE_GRID_3D(GRID,RES,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = RES; \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = 0; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE)));

#define CUDA_CREATE_GRID_1D_CPY(GRID,SIZE_,COUNT,SRC,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = make_uint3(SIZE_, 1, 1); \
	GRID.size = SIZE_; \
	GRID.count = COUNT; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE))); \
	checkCudaErrors(cudaMemcpy((void*)GRID.grid, (void*)SRC, (size_t)GRID.count*sizeof(ELETYPE), cudaMemcpyHostToDevice));

#define CUDA_CREATE_GRID_3D_CPY_S(GRID,SIZEX,SIZEY,SIZEZ,COUNT,SRC,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = make_uint3(SIZEX, SIZEY, SIZEZ); \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = COUNT; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE))); \
	checkCudaErrors(cudaMemcpy((void*)GRID.grid, (void*)SRC, (size_t)GRID.count*sizeof(ELETYPE), cudaMemcpyHostToDevice));

#define CUDA_CREATE_GRID_3D_CPY(GRID,RES,COUNT,SRC,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = RES; \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = COUNT; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE))); \
	checkCudaErrors(cudaMemcpy((void*)GRID.grid, (void*)SRC, (size_t)GRID.count*sizeof(ELETYPE), cudaMemcpyHostToDevice));

#define CUDA_CREATE_GRID_1D_SET(GRID,SIZE_,COUNT,VALUE,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = make_uint3(SIZE_, 1, 1); \
	GRID.size = SIZE_; \
	GRID.count = COUNT; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE))); \
	checkCudaErrors(cudaMemset((void*)GRID.grid, VALUE, (size_t)GRID.count*sizeof(ELETYPE)));

#define CUDA_CREATE_GRID_3D_SET_S(GRID,SIZEX,SIZEY,SIZEZ,COUNT,VALUE,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = make_uint3(SIZEX, SIZEY, SIZEZ); \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = COUNT; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE))); \
	checkCudaErrors(cudaMemset((void*)GRID.grid, VALUE, (size_t)GRID.count*sizeof(ELETYPE)));

#define CUDA_CREATE_GRID_3D_SET(GRID,RES,COUNT,VALUE,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = RES; \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = COUNT; \
	checkCudaErrors(cudaMalloc((void**)&GRID.grid, (size_t)GRID.size*sizeof(ELETYPE))); \
	checkCudaErrors(cudaMemset((void*)GRID.grid, VALUE, (size_t)GRID.count*sizeof(ELETYPE)));

#define SAFE_CUDA_FREE(PTR) safeCudaFree((void**)&PTR)
#define SAFE_CUDA_FREE_GRID(G) safeCudaFree((void**)&G.grid);
#define CUDA_DESTROY_GRID(G) SAFE_CUDA_FREE_GRID(G)\
	INITGRID_ZERO(G)

#define CUDA_GRID_CPY(GRID,SRC,COUNT,ELETYPE)\
	checkCudaErrors(cudaMemcpy((void*)GRID.grid, (void*)SRC, COUNT*sizeof(ELETYPE), cudaMemcpyHostToDevice)); \
	GRID.count = COUNT;

#define CUDA_GRID_SET(GRID,VALUE,COUNT,ELETYPE)\
	checkCudaErrors(cudaMemset((void*)GRID.grid, VALUE, COUNT*sizeof(ELETYPE))); \
	GRID.count = COUNT;

#define CUDA_MALLOC(PTR,SIZE_,ELETYPE)\
	checkCudaErrors(cudaMalloc((void**)&PTR, SIZE_*sizeof(ELETYPE)));

#define CUDA_CPY_TO_HOST(PTR,SRC,COUNT,ELETYPE)\
	checkCudaErrors(cudaMemcpy((void*)PTR, (void*)SRC, COUNT*sizeof(ELETYPE), cudaMemcpyDeviceToHost)); \

#define CUDA_SAFE_FREE(PTR)\
	if (PTR)\
{checkCudaErrors(cudaFree(PTR)); PTR = 0; }

#define CREATE_GRID_1D(GRID,SIZE_,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = make_uint3(SIZE_, 1, 1); \
	GRID.size = SIZE_; \
	GRID.count = 0; \
	GRID.grid = (ELETYPE*)malloc((size_t)GRID.size*sizeof(ELETYPE));

#define CREATE_GRID_3D_S(GRID,SIZEX,SIZEY,SIZEZ,ELETYPE)	GRID.grid=0;	\
	GRID.resolution = make_uint3(SIZEX, SIZEY, SIZEZ); \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = 0; \
	GRID.grid = (ELETYPE*)malloc((size_t)GRID.size*sizeof(ELETYPE));

#define CREATE_GRID_3D(GRID,RES,ELETYPE)	GRID.grid=0;	\
	GRID.resolution =RES; \
	GRID.size = GRID.resolution.x*GRID.resolution.y*GRID.resolution.z; \
	GRID.count = 0; \
	GRID.grid = (ELETYPE*)malloc((size_t)GRID.size*sizeof(ELETYPE));

#define SAFE_FREE(PTR) safeFree((void**)&PTR)
#define SAFE_FREE_GRID(G) safeFree((void**)&G.grid);
#define DESTROY_GRID(G) SAFE_FREE_GRID(G)\
						INITGRID_ZERO(G)

#define GRID_SET(GRID,VALUE,COUNT,ELETYPE)\
	memset((void*)GRID.grid, VALUE, COUNT*sizeof(ELETYPE)); \
	GRID.count = COUNT;

#define COPY_DEVICE_TO_HOST(H,D,SIZE_,ELETYPE) checkCudaErrors(cudaMemcpy(H, D,\
	SIZE_*sizeof(ELETYPE), cudaMemcpyDeviceToHost));

#define COPY_DEVICE_TO_HOST_GRID(HG,DG,ELETYPE) checkCudaErrors(cudaMemcpy(HG.grid, DG.grid,\
	DG.size*sizeof(ELETYPE), cudaMemcpyDeviceToHost));\
	HG.count = DG.size;

#define COPY_DEVICE_TO_HOST_GRID_C(HG,DG,COUNT,ELETYPE) checkCudaErrors(cudaMemcpy(HG.grid, DG.grid,\
	COUNT*sizeof(ELETYPE), cudaMemcpyDeviceToHost));\
	HG.count = COUNT;

#define COPY_HOST_TO_DEVICE_GRID(DG,HG,ELETYPE) checkCudaErrors(cudaMemcpy( DG.grid,HG.grid,\
	HG.size*sizeof(ELETYPE), cudaMemcpyHostToDevice));\
	DG.count = HG.size;

#define COPY_HOST_TO_DEVICE_GRID_C(DG,HG,COUNT,ELETYPE) checkCudaErrors(cudaMemcpy( DG.grid,HG.grid,\
	COUNT*sizeof(ELETYPE), cudaMemcpyHostToDevice));\
	DG.count = COUNT;