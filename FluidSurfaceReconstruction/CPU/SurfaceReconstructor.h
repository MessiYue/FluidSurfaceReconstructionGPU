#pragma once

#include <vector>

#include "MarchingCubes.h"

class SurfaceReconstructor : public MarchingCubes
{
public:

	SurfaceReconstructor();
	SurfaceReconstructor(ScalarGrid::ptr scalarGrid, float isoValue);
	virtual ~SurfaceReconstructor();

	//����ȫ�������ε����㻺����
	void generateTriangles();

	std::vector<fVector3> &getPosArray() { return mPosArray; }
	std::vector<fVector3> &getNorArray() { return mNorArray; }
	unsigned int getNumVertices()const { return mPosArray.size(); }

protected:
	std::vector<fVector3> mPosArray;
	std::vector<fVector3> mNorArray;

};
