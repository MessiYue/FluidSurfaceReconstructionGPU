#pragma once

#include <vector>
#include <string>

#include "Defines.h"
#include "../CPU/MathUtils.h"

class Utils
{
public:
	
	//! read particles' xyz of position from given .xyz file.
	static std::vector<SimpleParticle> readParticlesFromXYZ(
		const std::string &path,
		GridInfo *spatialGridInfo,
		float &particleRadius);

	//! save to .obj file.
	static void writeTrianglesToObj(
		const std::vector<Triangle> &triangles,
		const std::string &path);
	
private:
	Utils() = default;
	~Utils() = default;

};

