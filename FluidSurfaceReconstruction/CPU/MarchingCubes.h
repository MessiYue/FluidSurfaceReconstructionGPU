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

	//! 由index3D顶点所在的cube获取其产生的三角形及三角形个数
	//! index3D		三维顶点索引
	//! trianlges	存放当前三维顶点索引对应的cube产生的三角形
	//! triCount		三角形数量
	void getTriangles(const iVector3& index3D, Triangle triangles[5], int& triCount)const;

protected:

	//! 计算targetVal位于val0与val1间的百分点
	float getOffsetFac(float val0, float val1, float targetVal)const;

	//! 由偏移因子及两向量值插值出第三向量值
	fVector3 getOffsetVector(const fVector3& v0, const fVector3& v1, float offsetFac)const;

	//! 计算等值面上某点的梯度，这里忽略了分母
	fVector3 getGradient(const iVector3& index3D)const;


protected:
	float mIsoValue;
	ScalarGrid::ptr mScalarGrid;

};