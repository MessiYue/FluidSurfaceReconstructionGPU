#pragma once

#include "ReconstructorGPU.h"

class ReconstructorGPUOursZB05 : public ReconstructorGPU
{
public:
	typedef std::shared_ptr<ReconstructorGPUOursZB05> ptr;

	ReconstructorGPUOursZB05(const std::string &directory, const std::string &filePattern, unsigned int from, unsigned int to);

	virtual ~ReconstructorGPUOursZB05();

	virtual std::string getAlgorithmType();
	
protected:

	virtual void onBeginFrame(unsigned int frameIndex);

	virtual void onFrameMove(unsigned int frameIndex);

	virtual void onEndFrame(unsigned int frameIndex);

	//! finalization.
	virtual void onInitialization() override;

	//! initialization.
	virtual void onFinalization() override;

	//! estimation of surface vertices.
	void estimationOfSurfaceVertices();

	//! compactation of surface vertices.
	void compactationOfSurfaceVertices();

	//! calculation of scalar field grid with compacted array.
	void computationOfScalarFieldGrid();

};
