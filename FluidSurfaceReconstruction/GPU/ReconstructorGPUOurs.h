#pragma once

#include <string>
#include <vector>

#include "Timer.h"
#include "Defines.h"
#include "ReconstructorGPU.h"

class ReconstructorGPUOurs : public ReconstructorGPU
{
public:

	typedef std::shared_ptr<ReconstructorGPUOurs> ptr;

	ReconstructorGPUOurs(
		const std::string &directory,
		const std::string &filePattern,
		unsigned int from, unsigned int to);

	virtual ~ReconstructorGPUOurs() = default;

protected:
	virtual void onBeginFrame(unsigned int frameIndex) override;
	virtual void onFrameMove(unsigned int frameIndex) override;
	virtual void onEndFrame(unsigned int frameIndex) override;

	virtual void onInitialization() override;
	virtual void onFinalization() override;

	virtual void saveFluidSurfaceObjToFile(unsigned int frameIndex) override;
	virtual void getConfiguration(unsigned int frameIndex) override;

protected:

	//! detection of valid surface cubes.
	void detectionOfValidSurfaceCubes();
	//! compactation of valid surface cubes.
	void compactationOfValidSurafceCubes();
	//! generation of triangles for surface using marching cube algorithm.
	void generationOfSurfaceMeshUsingMC();

protected:
	uint mNumSurfaceMeshVertices;							//! number of vertices per cell.

	//! gpu arrays or grids.
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

};