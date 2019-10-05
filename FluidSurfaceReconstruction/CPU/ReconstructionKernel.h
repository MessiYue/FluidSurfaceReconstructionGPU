#pragma once

class ReconstructionKernel
{
public:
	static float kernel1(float distSq, float effectiveRadiusSq);
	static float kernel2(float distSq, float effectiveRadiusSq);
	static float kernel3(float distSq, float effectiveRadiusSq);
	static double kernel4(float distSq, float effectiveRadiusSq);

	static float weightFunc(float distSq, float effectiveRadiusSq);

	static double kernelZB05Sharp(double distSq, double EffectiveRadiusSq);
	static double kernelZB05Smooth(double distSq, double effectiveRadiusSq);

	static double kernelZB05SmoothGradFast(double dist, double effR);
	static double kernelZB05SmoothGrad(double distSq, double effectiveRadiusSq);

private:
	ReconstructionKernel() = default;
	~ReconstructionKernel() = default;

};
