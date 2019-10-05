
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "GPU/Utils.h"
#include "GPU/SurfaceReconstructorCUDA.h"

#define PATH "C:/Users/ywc/Desktop/FluidSimRet/DamBreakingPciSph/frame_000060.xyz"
#define TARGET "C:/Users/ywc/Desktop/FluidSimRet/DamBreakingPciSph/frame_000060.obj"


int main()
{
	SurfaceReconstructorCUDA::ptr reconstructor =
		std::shared_ptr<SurfaceReconstructorCUDA>(new SurfaceReconstructorCUDA());
	reconstructor->onInitialize(PATH);
	reconstructor->onFrameMove();
	std::vector<Triangle> triangles = reconstructor->getTriangles();
	Utils::writeTrianglesToObj(triangles, TARGET);
	reconstructor->onDestory();

    return 0;
}
