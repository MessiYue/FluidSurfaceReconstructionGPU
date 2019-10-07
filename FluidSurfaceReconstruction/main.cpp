
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "GPU/ReconstructorGPUOurs.h"


int main()
{
	ReconstructorGPUOurs::ptr reconstructor =
		std::shared_ptr<ReconstructorGPUOurs>(new ReconstructorGPUOurs(
			"C:/Users/ywc/Desktop/FluidSimRet/DamBreakingPciSph/",
			"frame_%06d", 0, 120));

	reconstructor->reconstruct();

    return 0;
}
