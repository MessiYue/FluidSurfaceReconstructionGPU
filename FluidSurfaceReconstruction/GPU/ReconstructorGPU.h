#pragma once

#include <string>
#include <vector>

#include "Timer.h"
#include "Defines.h"
#include "Kernel.cuh"

class ReconstructorGPU
{
public:

	typedef std::shared_ptr<ReconstructorGPU> ptr;

	ReconstructorGPU(const std::string &directory, const std::string &filePattern, unsigned int from, unsigned int to);

	virtual ~ReconstructorGPU();

	virtual void reconstruct();

	SimParam getSimulationParameters() const;

	unsigned int getNumberOfParticles() const;

	std::vector<double> getTimeConsumingSequence() const;

	virtual std::string getAlgorithmType() = 0;
	
protected:

	virtual void onBeginFrame(unsigned int frameIndex);

	virtual void onFrameMove(unsigned int frameIndex) = 0;

	virtual void onEndFrame(unsigned int frameIndex);

protected:

	//! read the particles from given file.
	std::vector<SimpleParticle> readParticlesFromFile(unsigned int frameIndex);

	//! spatial hashing grid building.
	void spatialHashingGridBuilding(const std::vector<SimpleParticle> &particles);

	//! save the surface .obj file.
	void saveFluidSurfaceObjToFile(unsigned int frameIndex);

	//! save the times record to file.
	void saveTimeConsumingRecordToFile();

	//! finalization.
	void onInitialization();

	//! initialization.
	void onFinalization();

protected:
	Timer::ptr mTimer;										//! timer for recording.
	std::string mFilePattern;								//! particles' file name's pattern.
	std::string mFileDirectory;								//! particles' file directory.
	unsigned int mFrameFrom, mFrameTo;						//! frames in [from, to).
	std::vector<double> mTimeConsuming;						//! times consumed for each frame.

	SimParam mSimParam;										//! simulation parameters.
	GridInfo mSpatialGridInfo;								//! spatial hashing grid information.
	GridInfo mScalarFieldGridInfo;							//! scalar field grid information.

	//! number of things.
	uint mNumParticles;										//! number of particles.
	uint mNumSurfaceVertices;								//! number of surface vertices.
	uint mNumValidSurfaceCubes;								//! number of valid surface cubes.
	uint mNumSurfaceMeshVertices;							//! number of valid surface vertices?????

	//! gpu arrays or grids.

	VertexGrid mDeviceScalarFieldGrid;						//! scalar field grid.
	ParticleArray mDeviceParticlesArray;					//! particles array.
	ParticleIndexInfoGrid mDeviceCellParticleIndexArray;	//! particles' start index and end index for each cell of grid.

	DensityGrid mDeviceDensityGrid;							//! virtual denisty field grid.
	IsSurfaceGrid mDeviceIsSurfaceGrid;						//! whether the vertex is in surface region or not.
	IsSurfaceGrid mDeviceIsSurfaceGridScan;					//! exclusive prefix sum of mDeviceIsSurfaceGrid.
	SurfaceVertexIndexArray mDeviceSurfaceVertexIndexArray;	//! compacted surface vertices' indices array.
	IsValidSurfaceGrid mDeviceIsValidSurfaceGrid;			//! whether the cube is valid or not.
	IsValidSurfaceGrid mDeviceIsValidSurfaceGridScan;		//! exculsive prefix sum of mDeviceIsValidSurfaceGrid.
	NumVerticesGrid mDeviceNumVerticesGrid;					//! number of each cube's vertices.
	NumVerticesGrid mDeviceNumVerticesGridScan;				//! exclusive prefix sum of mDeviceNumVerticesGrid.
	ValidSurfaceIndexArray mDeviceValidSurfaceIndexArray;	//! compacted valid surface cubes array.

	//! mesh's vertices and normals.
	Float3Grid mDeviceVertexArray;							//! vertex array of triangles.
	Float3Grid mDeviceNormalArray;							//! normal array of triangles.

	//! auxiliary cuda textures for marching cube algorithm.
	uint* mDeviceEdgeTable;									//! size=256, value=low 12 bits record intersection situation of 12 edges
	int* mDeviceEdgeIndicesOfTriangleTable;					//! size=256*16, value=each kind of situations intersected edges' number
	uint* mDeviceNumVerticesTable;							//! size=256, value=number of vertices for each situation.
	uint* mDeviceVertexIndicesOfEdgeTable;					//! size=12*2, value=each edge's start point and end point.

};