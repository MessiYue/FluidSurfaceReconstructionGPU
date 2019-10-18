
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

	ReconstructorGPUOurYu13::ptr reconstructor =
		std::shared_ptr<ReconstructorGPUOurYu13>(new ReconstructorGPUOurYu13(
			"C:/Users/ywc/Desktop/FluidSimRet/DamBreakingPciSph/",
			"frame_%06d", 20, 21));

	reconstructor->setOutputVisualizeFile(true);
	reconstructor->reconstruct();

    return 0;
}
