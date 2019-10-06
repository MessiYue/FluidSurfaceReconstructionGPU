#pragma once

#include "ReconstructorGPU.h"

class ReconstructorGPUOurs : public ReconstructorGPU
{
public:
	ReconstructorGPUOurs(const std::string &directory, const std::string &filePattern, unsigned int from, unsigned int to);

	virtual ~ReconstructorGPUOurs();

	virtual std::string getAlgorithmType();
	
protected:

	virtual void onBeginFrame(unsigned int frameIndex);

	virtual void onFrameMove(unsigned int frameIndex);

	virtual void onEndFrame(unsigned int frameIndex);

	//! estimation of surface vertices.
	void estimationOfSurfaceVertices();

	//! compactation of surface vertices.
	void compactationOfSurfaceVertices();

	//! calculation of scalar field grid with compacted array.
	void computationOfScalarFieldGrid();

	//! detection of valid surface cubes.
	void detectionOfValidSurfaceCubes();

	//! compactation of valid surface cubes.
	void compactationOfValidSurafceCubes();

	//! generation of triangles for surface using marching cube algorithm.
	void generationOfSurfaceMeshUsingMC();

private:


};
