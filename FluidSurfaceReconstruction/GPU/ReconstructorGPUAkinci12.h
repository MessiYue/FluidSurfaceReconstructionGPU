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


protected:
	virtual void onBeginFrame(unsigned int frameIndex) override;
	virtual void onFrameMove(unsigned int frameIndex) override;
	virtual void onEndFrame(unsigned int frameIndex) override;

	virtual void onInitialization() override;
	virtual void onFinalization() override;

	virtual void saveFluidSurfaceObjToFile(unsigned int frameIndex) override;

private:
	
	std::vector<float3> mVertexArray;							//! vertex array of surface mesh.
	std::vector<float3> mNormalArray;							//! normal array of surface mesh.

	IsSurfaceGrid mDeviceIsSurfaceGrid;							//! whether the vertex is in surface region or not.
	SurfaceVerticesIndexArray mDeviceSurfaceVerticesIndexArray;	//! compacted surface vertices' indices array.
	IsValidSurfaceGrid mDeviceIsValidSurfaceGrid;				//! whether the cube is valid or not.

};
