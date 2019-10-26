
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "GPU/ReconstructorGPUOursZB05.h"
#include "GPU/ReconstructorGPUOursYu13.h"

int main()
{
	//ReconstructorGPUOursZB05::ptr reconstructor =
	//	std::shared_ptr<ReconstructorGPUOursZB05>(new ReconstructorGPUOursZB05(
	//		"C:/Users/ywc/Desktop/FluidSimRet/DamBreakingPciSph/",
	//		"frame_%06d", 20, 21));

	ReconstructorGPUOursYu13::ptr reconstructor =
		std::shared_ptr<ReconstructorGPUOursYu13>(new ReconstructorGPUOursYu13(
			"C:/Users/ywc/Desktop/FluidSimRet/DamBreakingPciSph/",
			"frame_%06d", 20, 21));

	reconstructor->setOutputVisualizeFile(true);
	reconstructor->reconstruct();

    return 0;
}
