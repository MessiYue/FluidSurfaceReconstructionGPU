#include "ScalarGrid.h"

void ScalarGrid::reset(float val, bool isNorCal)
{
	for (int i = 0; i < (int)mVerticeDatas.size(); i++)
	{
		GridVertex* vertex = &mVerticeDatas[i];
		if (vertex != nullptr)
		{
			vertex->value = val;
			vertex->isNormalCal = isNorCal;
		}
	}
}

void ScalarGrid::resetE(float val, bool isNorCal)
{
	for (int i = 0; i < (int)mVerticeDatas.size(); i++)
	{
		GridVertex* vertex = &mVerticeDatas[i];
		if (vertex != nullptr)
		{
			vertex->value = val;
			vertex->isNormalCal = isNorCal;
			vertex->isNearSurface = false;
		}
	}
}

float ScalarGrid::getValue(int index1D)
{
	if (index1D >= 0 && index1D < (int)mVerticeDatas.size())
		return mVerticeDatas[index1D].value;
	return 0.f;
}

bool ScalarGrid::getValue(int index1D, float & _value)
{
	if (index1D >= 0 && index1D < (int)mVerticeDatas.size())
	{
		_value = mVerticeDatas[index1D].value;
		return true;
	}
	_value = 0.f;
	return false;
}

float ScalarGrid::getValue(const iVector3 & index3D)
{
	if (index3D.x >= 0 && index3D.x <= mGridRes.x &&
		index3D.y >= 0 && index3D.y <= mGridRes.y &&
		index3D.z >= 0 && index3D.z <= mGridRes.z)
		return mVerticeDatas[index3DTo1D(index3D)].value;
	return 0.f;
}

bool ScalarGrid::getValue(const iVector3 & index3D, float & _value)
{
	if (index3D.x >= 0 && index3D.x <= mGridRes.x &&
		index3D.y >= 0 && index3D.y <= mGridRes.y &&
		index3D.z >= 0 && index3D.z <= mGridRes.z)
	{
		_value = mVerticeDatas[index3DTo1D(index3D)].value;
		return true;
	}
	_value = 0.f;
	return false;
}

fVector3 ScalarGrid::getNormal(int index1D) const
{
	if (index1D >= 0 && index1D < (int)mVerticeDatas.size())
		return mVerticeDatas[index1D].normal;
	return fVector3(0.f, 0.f, 0.f);
}

fVector3 ScalarGrid::getNormal(const iVector3& index3D) const
{
	if (index3D.x >= 0 && index3D.x <= mGridRes.x &&
		index3D.y >= 0 && index3D.y <= mGridRes.y &&
		index3D.z >= 0 && index3D.z <= mGridRes.z)
		return mVerticeDatas[index3DTo1D(index3D)].normal;
	return fVector3(0.f, 0.f, 0.f);
}

const fVector3 ScalarGrid::calcNormal(const iVector3& index3D)
{
	int i = index3D.x;
	int j = index3D.y;
	int k = index3D.z;
	// central difference.
	float gradX = getValue(iVector3(i - 1, j, k)) - getValue(iVector3(i + 1, j, k));
	float gradY = getValue(iVector3(i, j - 1, k)) - getValue(iVector3(i, j + 1, k));
	float gradZ = getValue(iVector3(i, j, k - 1)) - getValue(iVector3(i, j, k + 1));
	return fVector3(gradX, gradY, gradZ);
}

void ScalarGrid::updateNormals()
{
	for (int z = 0; z < mGridRes.z + 1; z++)
	{
		for (int y = 0; y < mGridRes.y + 1; y++)
		{
			for (int x = 0; x < mGridRes.x + 1; x++)
			{
				iVector3 index3D(x, y, z);
				int index1D = index3DTo1D(index3D);
				mVerticeDatas[index1D].normal = calcNormal(index3D);
				mVerticeDatas[index1D].normal.normalize();
			}
		}
	}
}

void ScalarGrid::updatePositions()
{
	for (int z = 0; z < mGridRes.z + 1; z++)
	{
		for (int y = 0; y < mGridRes.y + 1; y++)
		{
			for (int x = 0; x < mGridRes.x + 1; x++)
			{
				int index1D = index3DTo1D(iVector3(x, y, z));
				mVerticeDatas[index1D].position = 
					fVector3(x*mCellSize, y*mCellSize, z*mCellSize) + mGridBox.getMin();
			}
		}
	}
}
