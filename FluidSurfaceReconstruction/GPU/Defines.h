#pragma once

#include <cstdlib>
#include <iostream>
#include <vector_types.h>
#include <vector_functions.h>

#define TRUE_ 1
#define FALSE_ 0

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long long ulonglong;

template<typename T>
struct Grid 
{
	T* grid;
	ulonglong size;
	ulonglong count;
	uint3 resolution;
};

struct IndexInfo 
{
	uint start, end;
};

struct SimpleParticle 
{
	float3 pos;
};

struct ParticleAk 
{
	float3 pos;
	float density;
};

struct SimpleVertex 
{
	float value;
};

struct VertexAkinci
{
	float value;
	uint indexInScGrid;
};

struct GridInfo
{
	uint3 resolution;
	float3 minPos;
	float3 maxPos;
	float cellSize;
};

typedef Grid<int> IntGrid;
typedef Grid<uint> UintGrid;
typedef Grid<float> FloatGrid;
typedef Grid<float3> Float3Grid;

typedef Grid<float> DensityGrid;
typedef Grid<uint> IsSurfaceGrid;

typedef Grid<SimpleParticle> ParticleArray;
typedef Grid<ParticleAk> ParticleAkArray;
typedef Grid<IndexInfo> ParticleIndexInfoGrid;

typedef Grid<SimpleVertex>  VertexGrid;							
typedef Grid<VertexAkinci> VerAkGrid;							

typedef Grid<uint> NumVerticesGrid;								// number of vertices.
typedef Grid<uint> IsValidSurfaceGrid;							// whethe valid or not for cells.
typedef Grid<uint> SurfaceVertexIndexArray;						// compacted surface vertices' indices.
typedef Grid<uint> ValidSurfaceIndexArray;						// compacted valid surface cells' indices.

inline std::ostream& operator<<(std::ostream& os, const VertexAkinci& v)
{
	os << "IdxSc=" << v.indexInScGrid << " v=" << v.value;
	return os;
}

inline void safeFree(void** ptr)
{
	if (ptr == 0 || *ptr == 0)
		return;
	free(*ptr);
	*ptr = 0;
}

inline uint index31(uint3 index3, uint3 res)
{
	return index3.z*res.x*res.y + index3.y*res.x + index3.x;
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