#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#include "Defines.h"

struct SimParam
{
	float smoothingRadius;				// smoothing radius.
	float particleMass;					// particle mass.
	float particleRadius;				// particle radius.
	float coeSurfaceTension;			// surface tension coefficient.
	float simScale;
	float sfParticleGradThreshold;		// gradient threshold of surface particles.

	float effR;							// effective radius.
	float effRSq;						// square of effective radius.
	float isoValue;						// contour value
	float worldParRadius;				
	uint scSpGridResRatio;

};

__constant__ SimParam simParam;

extern "C"
{
	void initSimParam(SimParam *params);

	uint ThrustExclusiveScanWrapper(uint* output, uint* input, uint numElements);
}

//extern "C" { void initSimParam(SimParam simParam); }
//
//extern "C" uint ThrustExclusiveScanWrapper(uint* output, uint* input, uint numElements);

inline __device__ 
float3 computeGradColorField(float massJ, float densityJ, float3 ri_rj, float dist, float sr)
{
	// massJ -> mass of particle j.
	// densityJ -> density of particle j.
	// ri_rj -> ri - rj.
	// dist -> |ri - rj|.
	// sr -> smoothing radius.
	float h2_r2 = sr * sr - dist * dist;
	float3 grad = ri_rj * simParam.coeSurfaceTension * h2_r2 * h2_r2;
	return grad * massJ / densityJ;
}

inline __device__ 
float3 computeGradColorFieldFast(float massJ, float densityJ, float3 ri_rj, float distSq, float srSq)
{
	float h2_r2 = srSq - distSq;
	float3 grad = ri_rj * simParam.coeSurfaceTension * h2_r2 * h2_r2;
	return grad * massJ / densityJ;
}

inline __device__ 
float kernelZB05Smooth(float distSq, float effetiveRadiusSq)
{
	if (distSq >= effetiveRadiusSq)
		return 0;
	return pow(1.0f - distSq / effetiveRadiusSq, 3);
}

inline __host__ __device__ 
float weightFunc(float distSq, float EffectiveRadiusSq)
{
	if (distSq > EffectiveRadiusSq)
		return 0.f;
	float d2DR2 = distSq / EffectiveRadiusSq * 0.5f;
	float d2DR2_2 = d2DR2 * d2DR2;
	return d2DR2_2 - d2DR2 + 0.25f;
}

inline __host__ __device__ 
float weightFunc2(float distSq, float EffectiveRadiusSq)
{
	if (distSq > EffectiveRadiusSq)
		return 0.f;
	float t2 = distSq / EffectiveRadiusSq * 0.5f;
	float t4 = t2 * t2;
	float t8 = t4 * t4;
	return t8 - t4 + 0.1875f;
}

#endif
