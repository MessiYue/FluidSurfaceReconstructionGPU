
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "GPU/ReconstructorGPUOursZB05.h"


int main()
{
	ReconstructorGPUOursZB05::ptr reconstructor =
		std::shared_ptr<ReconstructorGPUOursZB05>(new ReconstructorGPUOursZB05(
			"C:/Users/ywc/Desktop/FluidSimRet/DamBreakingPciSph/",
			"frame_%06d", 0, 30));

	reconstructor->reconstruct();

    return 0;
}
