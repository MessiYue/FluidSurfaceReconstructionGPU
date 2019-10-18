#pragma once

#include "ReconstructorGPU.h"

class ReconstructorGPUOurYu13 final : public ReconstructorGPU
{
public:
	typedef std::shared_ptr<ReconstructorGPUOurYu13> ptr;

	ReconstructorGPUOurYu13(const std::string &directory, const std::string &filePattern,
		unsigned int from, unsigned int to);

	virtual ~ReconstructorGPUOurYu13();

	virtual std::string getAlgorithmType();

protected:
	virtual void onBeginFrame(unsigned int frameIndex) override;

	virtual void onFrameMove(unsigned int frameIndex) override;

	virtual void onEndFrame(unsigned int frameIndex) override;

	virtual void onInitialization() override;

	virtual void onFinalization() override;

	virtual void saveMiddleDataToVisFile(unsigned int frameIndex) override;

private:

	//! calculation of mean pos and smoothed pos for particles.
	void calculationOfMeanAndSmoothedParticles();

	//! estimation of surface vertices and surface particles.
	void estimationOfSurfaceVerticesAndParticles();

	//! compactation of surface particles.
	void compactationOfSurfaceVerticesAndParticles();

	//! calculation of transform matrices for each surface particle.
	void calculationOfTransformMatricesForParticles();

	//! calculation of scalar field grid.
	void computationOfScalarFieldGrid();
	
private:

	uint mNumSurfaceParticles;										//! number of surface particles.
	ParticleArray mDeviceParticlesMean;								//! mean positions of particles.
	ParticleArray mDeviceParticlesSmoothed;							//! smoothed positions of particles.
	NumSurfaceParticlesGrid mDeviceNumSurfaceParticlesGrid;			//! number of surface particles for each cell.
	NumSurfaceParticlesGrid mDeviceNumSurfaceParticlesGridScan;		//! number of surface particles for each cell.
	SurfaceParticlesIndexArray mDeviceSurfaceParticlesIndexArray;	//! surface particles' indices.
	MatrixArray mDeviceSVDMatricesArray;							//! transform matrices for anisotropic kernel.

};
