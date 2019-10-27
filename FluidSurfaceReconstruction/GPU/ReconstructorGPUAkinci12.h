#pragma once

#include <string>
#include <vector>
#include "Timer.h"
#include "Defines.h"
#include "ReconstructorGPU.h"

class ReconstructorGPUAkinci12 final : public ReconstructorGPU
{
public:

	typedef std::shared_ptr<ReconstructorGPUAkinci12> ptr;

	ReconstructorGPUAkinci12(
		const std::string &directory,
		const std::string &filePattern,
		unsigned int from, unsigned int to);

	virtual ~ReconstructorGPUAkinci12() = default;

	virtual std::string getAlgorithmType();


protected:
	virtual void onBeginFrame(unsigned int frameIndex) override;
	virtual void onFrameMove(unsigned int frameIndex) override;
	virtual void onEndFrame(unsigned int frameIndex) override;

	virtual void onInitialization() override;
	virtual void onFinalization() override;

	virtual void saveMiddleDataToVisFile(unsigned int frameIndex) override;
	virtual void saveFluidSurfaceObjToFile(unsigned int frameIndex) override;

private:
	
	//! surface particles extraction using color field method.
	void extractionOfSurfaceParticles();
	//! surface vertices extraction using surface particles.
	void extractionOfSurfaceVertices();
	//! compactation of surface vertices.
	void compactationOfSurfaceVertices();
	//! scalar field computation for narrow-band area.
	void computationOfScalarFieldGrid();
	//! detection of valid surface vertices.
	void detectionOfValidSurfaceCubes();
	//! generation of triangles for surface using marching cube algorithm.
	void generationOfSurfaceMeshUsingMC();

private:
	
	std::vector<fVector3> mVertexArray;							//! vertex array of surface mesh.
	std::vector<fVector3> mNormalArray;							//! normal array of surface mesh.

	IsSurfaceGrid mDeviceIsSurfaceGrid;							//! whether the vertex is in surface region or not.
	SurfaceVerticesIndexArray mDeviceSurfaceVerticesIndexArray;	//! compacted surface vertices' indices array.
	IsValidSurfaceGrid mDeviceIsValidSurfaceGrid;				//! whether the cube is valid or not.
	std::vector<uint> mHostSurfaceVerticesIndexArray;
	std::vector<uint> mValidSurfaceCubesIndexArray;

};
