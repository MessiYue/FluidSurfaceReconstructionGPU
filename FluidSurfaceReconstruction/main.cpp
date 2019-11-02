
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "GPU/ReconstructorGPUAkinci12.h"
#include "GPU/ReconstructorGPUOursZB05.h"
#include "GPU/ReconstructorGPUOursYu13.h"

int main()
{
	unsigned int testCount = 1;
	for (unsigned int i = 0; i < testCount; ++i)
	{
		ReconstructorGPUOursZB05::ptr reconstructor =
			std::shared_ptr<ReconstructorGPUOursZB05>(new ReconstructorGPUOursZB05(
				"C:/Users/ywc/Desktop/FluidSimRet/Hemisphere/",
				"frame_%06d", 0, 23));

		//ReconstructorGPUOursYu13::ptr reconstructor =
		//	std::shared_ptr<ReconstructorGPUOursYu13>(new ReconstructorGPUOursYu13(
		//		"C:/Users/ywc/Desktop/FluidSimRet/Hemisphere/",
		//		"frame_%06d", 22, 23));

		//ReconstructorGPUAkinci12::ptr reconstructor =
		//	std::shared_ptr<ReconstructorGPUAkinci12>(new ReconstructorGPUAkinci12(
		//		"C:/Users/ywc/Desktop/FluidSimRet/BunnyDrop/",
		//		"frame_%06d", 0, 23));

		reconstructor->setOutputMeshFile(false);
		reconstructor->setOutputConfigFile(true);
		reconstructor->setOutputVisualizeFile(false);
		reconstructor->reconstruct();
	}

    return 0;
}
