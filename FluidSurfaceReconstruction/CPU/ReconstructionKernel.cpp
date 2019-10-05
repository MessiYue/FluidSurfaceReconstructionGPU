#include "ReconstructionKernel.h"

#include <algorithm>

float ReconstructionKernel::kernel1(float distSq, float effectiveRadiusSq)
{
	if (distSq >= effectiveRadiusSq)
		return 0.f;
	return 1 - std::pow(std::sqrt(distSq / effectiveRadiusSq), 3);
}

float ReconstructionKernel::kernel2(float distSq, float effectiveRadiusSq)
{
	if (distSq >= effectiveRadiusSq)
		return 0.f;
	float t2 = distSq / effectiveRadiusSq;
	float t4 = t2 * t2;
	float t8 = t4 * t4;
	return 0.0625f * t8 - 0.25f * t4 + 0.1875f;
}

float ReconstructionKernel::kernel3(float distSq, float effectiveRadiusSq)
{
	if (distSq >= effectiveRadiusSq)
		return 0.f;
	return sqrt(effectiveRadiusSq / distSq) - 1;
}

double ReconstructionKernel::kernel4(float distSq, float effectiveRadiusSq)
{
	if (distSq >= effectiveRadiusSq)
		return 0.f;
	return distSq ? (effectiveRadiusSq / distSq - 1.0) : 1e20;
}

float ReconstructionKernel::weightFunc(float distSq, float effectiveRadiusSq)
{
	if (distSq >= effectiveRadiusSq)
		return 0.f;
	float d2DR2 = distSq / effectiveRadiusSq * 0.5f;
	float d2DR2_2 = d2DR2 * d2DR2;	
	return d2DR2_2 - d2DR2 + 0.25f;
}

double ReconstructionKernel::kernelZB05SmoothGradFast(double dist, double effR)
{
	if (dist >= effR)
		return 0;
	double s = dist / effR;
	double t = 1 - s * s;
	return -6 * s * t * t;
}

double ReconstructionKernel::kernelZB05Sharp(double distSq, double EffectiveRadiusSq)
{
	if (distSq >= EffectiveRadiusSq)
		return 0;
	return distSq ? (EffectiveRadiusSq / distSq - 1.0) : 1e20;
}

double ReconstructionKernel::kernelZB05Smooth(double distSq, double effectiveRadiusSq)
{
	if (distSq >= effectiveRadiusSq)
		return 0;
	return pow(1.0 - distSq / effectiveRadiusSq, 3);
}

double ReconstructionKernel::kernelZB05SmoothGrad(double distSq, double effectiveRadiusSq)
{
	if (distSq >= effectiveRadiusSq)
		return 0;
	double r = distSq / effectiveRadiusSq;
	double s = sqrt(r);
	double t = 1 - r;
	return -6 * s * t * t;
}
