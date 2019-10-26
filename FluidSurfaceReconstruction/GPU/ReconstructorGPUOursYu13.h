#pragma once

#include "ReconstructorGPUOurs.h"

class ReconstructorGPUOursYu13 final : public ReconstructorGPUOurs
{
public:
	typedef std::shared_ptr<ReconstructorGPUOursYu13> ptr;

	ReconstructorGPUOursYu13(const std::string &directory, const std::string &filePattern,
		unsigned int from, unsigned int to);

	virtual ~ReconstructorGPUOursYu13() = default;

	virtual std::string getAlgorithmType();

protected:
	virtual void onBeginFrame(unsigned int frameIndex) override;
	virtual void onFrameMove(unsigned int frameIndex) override;
	virtual void onEndFrame(unsigned int frameIndex) override;

	virtual void onInitialization() override;
	virtual void onFinalization() override;

	virtual void saveMiddleDataToVisFile(unsigned int frameIndex) override;

private:

	//! extraction of surface particles.
	void extractionOfSurfaceAndInvolveParticles();
	//! estimation of surface vertices and involve particles.
	void estimationOfSurfaceVertices();
	//! calculation of mean pos and smoothed pos for particles.
	void calculationOfMeanAndSmoothedParticles();
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
	NumInvolveParticlesGrid mDeviceNumInvolveParticlesGrid;			//! number of surface particles for each cell.
	NumInvolveParticlesGrid mDeviceNumInvolveParticlesGridScan;		//! number of surface particles for each cell.
	SurfaceParticlesIndexArray mDeviceInvolveParticlesIndexArray;	//! surface particles' indices.
	MatrixArray mDeviceSVDMatricesArray;							//! transform matrices for anisotropic kernel.

};
