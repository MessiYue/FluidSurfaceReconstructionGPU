#pragma once

#include <string>
#include <vector>

#include "Timer.h"
#include "Defines.h"

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

	void setOutputVisualizeFile(bool flag);
	
private:

	void beginFrame(unsigned int frameIndex);

	void frameMove(unsigned int frameIndex);

	void endFrame(unsigned int frameIndex);

	void initialization();

	void finalization();

protected:
	virtual void onBeginFrame(unsigned int frameIndex) = 0;

	virtual void onFrameMove(unsigned int frameIndex) = 0;

	virtual void onEndFrame(unsigned int frameIndex) = 0;

	virtual void saveMiddleDataToVisFile(unsigned int frameIndex) = 0;

	virtual void onInitialization() = 0;

	virtual void onFinalization() = 0;

protected:

	//! read the particles from given file.
	void readParticlesFromFile(unsigned int frameIndex,
		std::vector<ParticlePosition> &particles, std::vector<ScalarValue> &densities);

	//! spatial hashing grid building.
	void spatialHashingGridBuilding();

	//! save the surface .obj file.
	void saveFluidSurfaceObjToFile(unsigned int frameIndex);

	//! save the times record to file.
	void saveTimeConsumingRecordToFile();

	//! detection of valid surface cubes.
	void detectionOfValidSurfaceCubes();

	//! compactation of valid surface cubes.
	void compactationOfValidSurafceCubes();

	//! generation of triangles for surface using marching cube algorithm.
	void generationOfSurfaceMeshUsingMC();

protected:
	bool mSaveVisFile;										//! tag for saving visualization file(.vis)
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

	ScalarFieldGrid mDeviceScalarFieldGrid;					//! scalar field grid.
	ParticleArray mDeviceParticlesArray;					//! particles array.
	ScalarFieldGrid mDeviceParticlesDensityArray;			//! particle density array.
	ParticleIndexRangeGrid mDeviceCellParticleIndexArray;	//! particles' start index and end index for each cell of grid.

	DensityGrid mDeviceFlagGrid;							//! virtual denisty field grid.
	IsSurfaceGrid mDeviceSurfaceParticlesFlagGrid;			//! whether the particle is in surface region or not.
	IsSurfaceGrid mDeviceIsSurfaceGrid;						//! whether the vertex is in surface region or not.
	IsSurfaceGrid mDeviceIsSurfaceGridScan;					//! exclusive prefix sum of mDeviceIsSurfaceGrid.
	SurfaceVerticesIndexArray mDeviceSurfaceVerticesIndexArray;//! compacted surface vertices' indices array.
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