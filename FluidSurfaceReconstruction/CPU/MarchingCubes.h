#pragma once

#include "MathUtils.h"
#include "MathVector.h"
#include "ScalarGrid.h"
#include "MarchingCubesHelper.h"

class MarchingCubes
{
public:

	MarchingCubes() : MarchingCubes(nullptr, 0) {}

	MarchingCubes(ScalarGrid::ptr scalarGrid, float isoValue) :
		mScalarGrid(scalarGrid), mIsoValue(isoValue) {}

	virtual ~MarchingCubes() = default;

	float getIsoValue() const { return mIsoValue; }
	const ScalarGrid::ptr getScalarGrid() const { return mScalarGrid; }

	void setIsoValue(float isoValue) { mIsoValue = isoValue; }
	void setScalarGrid(ScalarGrid::ptr scalarGrid) { mScalarGrid = scalarGrid; }

	//! ��index3D�������ڵ�cube��ȡ������������μ������θ���
	//! index3D		��ά��������
	//! trianlges	��ŵ�ǰ��ά����������Ӧ��cube������������
	//! triCount		����������
	void getTriangles(const iVector3& index3D, Triangle triangles[5], int& triCount)const;

protected:

	//! ����targetValλ��val0��val1��İٷֵ�
	float getOffsetFac(float val0, float val1, float targetVal)const;

	//! ��ƫ�����Ӽ�������ֵ��ֵ����������ֵ
	fVector3 getOffsetVector(const fVector3& v0, const fVector3& v1, float offsetFac)const;

	//! �����ֵ����ĳ����ݶȣ���������˷�ĸ
	fVector3 getGradient(const iVector3& index3D)const;


protected:
	float mIsoValue;
	ScalarGrid::ptr mScalarGrid;

};