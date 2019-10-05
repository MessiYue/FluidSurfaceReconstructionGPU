#pragma once

#include "Defines.h"
#include "Kernel.cuh"
#include "../CPU/MathVector.h"
#include "../CPU/ScalarGrid.h"

class SurfaceReconstructorCUDA
{
public:

	typedef std::shared_ptr<SurfaceReconstructorCUDA> ptr;

	SurfaceReconstructorCUDA();
	virtual ~SurfaceReconstructorCUDA();

	void onInitialize(const std::string &path);
	void onFrameMove();
	void onDestory();

	//! -------------------------------Our method----------------------------------

	void estimateSurfaceVertex();					// estimation of surface vertices.
	void updateScalarGridValues();					// calculation of scalar field grid.
	void compactSurfaceVertex();					// compact surface vertices into continuous array.
	void updateScalarGridValuesCompacted();			// calculation of scalar field grid with compacted array.
	void detectValidSurfaceCubes();					// detection of valid surface cubes.
	void compactValidSurafceCubes();				// compactation of valid surface cubes.
	void generateTriangles();						// generation of triangles for surface.
	std::vector<Triangle> getTriangles();			// get triangles of surface.

protected:
	SimParam mSimParam;										// simulation parameters.

	//! auxiliary cuda arrays for scalar field calculation.
	VertexGrid d_VertexGrid;								// scalar field grid.

	uint mNumParticles;										// number of particles.
	DensityGrid d_DensityGrid;								// virtual denisty field grid.
	ParticleArray d_ParticleArray;							// particles array.
	ParticleIndexInfoGrid d_ParticleIndexInfoGrid;			// particles' indices for each cell of grid.
	SurfaceVertexIndexArray d_SurfaceVertexIndexArray;		// compacted surface vertices' indices array.
	IsSurfaceGrid d_IsSurfaceGrid;							// whether the vertex is in surface region or not.
	IsSurfaceGrid d_IsSurfaceGridScan;						// exculsive prefix sum of d_IsSurfaceGrid.
	uint mNumSurfaceVertex;									// num of surface vertices.

	IsValidSurfaceGrid d_IsValidSurfaceGrid;				// whether the cube is valid or not.
	IsValidSurfaceGrid d_IsValidSurfaceGridScan;			// exculsive prefix sum of d_IsValidSurfaceGrid.
	NumVerticesGrid d_numVerticesGrid;						// number of each cube's vertices.
	NumVerticesGrid d_numVerticesGridScan;					// exculsive prefix sum of d_numVerticesGrid.
	ValidSurfaceIndexArray d_validSurfaceIndexArray;		// compacted valid surface cubes array.
	uint mNumValidSurfaceCubes;								// number of valid surface cubes.
	uint mNumValidVertices;									// number of valid surface vertices?????

	Float3Grid d_posGrid;									// postions of triangles.
	Float3Grid d_norGrid;									// normals of triangles.

	// auxiliary cuda textures for marching cube algorithm.
	uint* d_edgeTable;										// size=256, value=low 12 bits record intersection situation of 12 edges
	int* d_edgeIndicesOfTriangleTable;						// size=256, value=each kind of situations intersected edges' number
	uint* d_numVerticesTable;								// size=256, value=number of vertices for each situation.
	uint* d_vertexIndicesOfEdgeTable;						// size=12*2, value=each edge's start point and end point.

	// cuda grid information
	GridInfo mSpGridInfo;									// spatial hashing grid information.
	GridInfo mScGridInfo;									// scalar field grid information.

	// create auxiliary textures.
	void createTextures();									// creation of auxiliary textures for marching cube.

};
