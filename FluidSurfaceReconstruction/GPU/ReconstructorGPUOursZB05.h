#pragma once

#include "ReconstructorGPU.h"

class ReconstructorGPUOursZB05 final : public ReconstructorGPU
{
public:
	typedef std::shared_ptr<ReconstructorGPUOursZB05> ptr;

	ReconstructorGPUOursZB05(const std::string &directory, const std::string &filePattern,
		unsigned int from, unsigned int to);

	virtual ~ReconstructorGPUOursZB05();

	virtual std::string getAlgorithmType();
	
protected:

	virtual void onBeginFrame(unsigned int frameIndex) override;

	virtual void onFrameMove(unsigned int frameIndex) override;

	virtual void onEndFrame(unsigned int frameIndex) override;

	virtual void onInitialization() override;

	virtual void onFinalization() override;

	virtual void saveMiddleDataToVisFile(unsigned int frameIndex) override;

private:

	//! estimation of surface vertices.
	void estimationOfSurfaceVertices();

	//! compactation of surface vertices.
	void compactationOfSurfaceVertices();

	//! calculation of scalar field grid with compacted array.
	void computationOfScalarFieldGrid();

};
