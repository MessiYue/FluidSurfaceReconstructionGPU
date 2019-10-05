#pragma once

#include <vector>

#include "MarchingCubes.h"

class SurfaceReconstructor : public MarchingCubes
{
public:

	SurfaceReconstructor();
	SurfaceReconstructor(ScalarGrid::ptr scalarGrid, float isoValue);
	virtual ~SurfaceReconstructor();

	//产生全部三角形到顶点缓冲区
	void generateTriangles();

	std::vector<fVector3> &getPosArray() { return mPosArray; }
	std::vector<fVector3> &getNorArray() { return mNorArray; }
	unsigned int getNumVertices()const { return mPosArray.size(); }

protected:
	std::vector<fVector3> mPosArray;
	std::vector<fVector3> mNorArray;

};
