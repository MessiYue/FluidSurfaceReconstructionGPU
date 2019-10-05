#pragma once

#include <vector>
#include <memory>
#include <string>

#include "Box3.h"
#include "MathVector.h"

template <typename T>
class GridInterface
{
public:

	GridInterface() = default;
	virtual ~GridInterface() = default;

	//! initialization.
	void resetFast(int value = 0);
	void init(const fBox3& gridBox, float cellSize);
	void init(const fBox3& gridBox, const iVector3& gridRes);

	//! get corresponding data.
	bool getVertexData(int index1D, T& data);
	virtual float getValue(const iVector3& index3D) = 0;
	virtual bool getValue(const iVector3& index3D, float& _value) = 0;

	//! get corresponding pointer of data.
	T* getVertexDataP(int index1D);
	T* getVertexDataP(const iVector3& index3D);
	bool getVertexData(const iVector3& index3D, T& data);

	//! index3D -> spatial position.
	fVector3 getPos(const iVector3& index3D) const;
	//! spatial position -> index3D.
	iVector3 getIndex3D(const fVector3& pos) const;
	//! spatial position -> index1D.
	int getIndex1D(const fVector3& pos) const;

	//! index3D -> index1D.
	int index3DTo1D(int x, int y, int z) const;
	int index3DTo1D(const iVector3& index3D) const;
	
	//! index1D -> index3D.
	iVector3 index1DTo3D(int index1D) const;

	//! 由立方体八个顶点中的最小一维索引获取8个一维索引
	void getCubeIndexes1D(const iVector3& min, int indexes[8]) const;
	//! 由立方体八个顶点中的最小三维索引获取8个三维索引
	void getCubeIndexes3D(const iVector3& min, iVector3 indexes[8]) const;

	//! 获取邻近（包括自身）27个顶点索引
	void getNeighborIndexes1D27(const iVector3& center, int indices[27]) const;
	void getNeighborIndexes3D27(const iVector3& center, iVector3 indices[27]) const;

	float getCellSize() const { return mCellSize; }
	const fBox3& getGridBox() const { return mGridBox; }
	const fVector3& getGridSize() const { return mGridSize; }
	const iVector3& getGridRes() const { return mGridRes; }

	void printDatasToFile(const std::string& fileName, int numElePerLine = 100)const;

protected:
	// cell length.
	float mCellSize;
	// bounding box.
	fBox3 mGridBox;
	// size of bounding box.
	fVector3 mGridSize;
	// resolution
	iVector3 mGridRes;
	// mGridFac=mGridRes/mGridSize.
	fVector3 mGridFac;
	// grid vertices.
	std::vector<T> mVerticeDatas;

};

//! -------------------------------------Definition------------------------------------

template<typename T>
inline void GridInterface<T>::resetFast(int value/*=0*/)
{
	memset(&mVerticeDatas[0], value, mVerticeDatas.size() * sizeof(T));
}

template<typename T>
inline void GridInterface<T>::init(const fBox3& gridBox, float cellSize)
{
	mGridBox = gridBox;
	mCellSize = cellSize;
	mGridSize = mGridBox.getSize();
	mGridRes = (mGridSize / mCellSize).toIntVector();

	/*//由于上面计算分辨率时对浮点进行截取操作，因此下面需要对mGridSize进行微调
	fVector3 adjustedGridSize=fVector3(mGridRes)*mCellSize;
	gridBox.setMax(gridBox.getMax()+adjustedGridSize-mGridSize);
	mGridSize=adjustedGridSize;*/

	mGridFac = fVector3(mGridRes) / mGridSize;
	mVerticeDatas.resize((mGridRes.x + 1)*(mGridRes.y + 1)*(mGridRes.z + 1));
}

template<typename T>
inline void GridInterface<T>::init(const fBox3& gridBox, const iVector3& gridRes)
{
	mGridBox = gridBox;
	mGridRes = gridRes;
	//if ((mGridRes.x & 1) == 0)
	//	mGridRes += 1;
	mGridSize = mGridBox.getSize();
	mCellSize = mGridSize.x / mGridRes.x;
	mGridFac = fVector3(mGridRes) / mGridSize;
	mVerticeDatas.resize((mGridRes.x + 1)*(mGridRes.y + 1)*(mGridRes.z + 1));
}

template<typename T>
inline fVector3 GridInterface<T>::getPos(const iVector3& index3D) const
{
	return fVector3(fVector3(index3D)*mCellSize) + mGridBox.getMin();
}

template<typename T>
inline iVector3 GridInterface<T>::getIndex3D(const fVector3& pos) const
{
	return ((pos - mGridBox.getMin()) / mCellSize).toIntVector();
}

template<typename T>
inline int GridInterface<T>::getIndex1D(const fVector3& pos) const
{
	return index3DTo1D(getIndex3D(pos));
}

template<typename T>
inline int GridInterface<T>::index3DTo1D(int x, int y, int z) const
{
	if (x<0 || x>mGridRes.x || y<0 || y>mGridRes.y || z<0 || z>mGridRes.z)
		return -1;
	return (z*(mGridRes.y + 1) + y)*(mGridRes.x + 1) + x;
}

template<typename T>
inline int GridInterface<T>::index3DTo1D(const iVector3 & index3D) const
{
	return index3DTo1D(index3D.x, index3D.y, index3D.z);
}

template<typename T>
inline iVector3 GridInterface<T>::index1DTo3D(int index1D) const
{
	iVector3 res;
	int xy = (mGridRes.x + 1) * (mGridRes.y + 1);
	res.z = index1D / xy;
	int mod = index1D % xy;
	res.y = mod / (mGridRes.x + 1);
	res.x = mod % (mGridRes.x + 1);
	return res;
}

template<typename T>
inline T* GridInterface<T>::getVertexDataP(int index1D)
{
	if (index1D >= 0 && index1D < (int)mVerticeDatas.size())
		return &mVerticeDatas[index1D];
	return nullptr;
}

template<typename T>
inline T* GridInterface<T>::getVertexDataP(const iVector3& index3D)
{
	return getVertexDataP(index3DTo1D(index3D));
}

template<typename T>
inline bool GridInterface<T>::getVertexData(const iVector3& index3D, T& data)
{
	return getVertexData(index3DTo1D(index3D), data);
}

template<typename T>
inline bool GridInterface<T>::getVertexData(int index1D, T & data)
{
	if (index1D >= 0 && index1D < (int)mVerticeDatas.size())
	{
		data = mVerticeDatas[index1D];
		return true;
	}
	return false;
}

template<typename T>
inline void GridInterface<T>::getCubeIndexes1D(const iVector3& min, int indexes[8]) const
{
	iVector3 indexes3D[8];
	getCubeIndexes3D(min, indexes3D);
	for (int i = 0; i < 8; i++)
		indexes[i] = index3DTo1D(indexes3D[i]);
}

template<typename T>
inline void GridInterface<T>::getCubeIndexes3D(const iVector3& min, iVector3 indexes[8]) const
{
	for (int i = 0; i < 8; i++)
	{
		const int* offset = cubeVerticesOffset[i];
		iVector3 offsetVec(offset[0], offset[1], offset[2]);
		indexes[i] = min + offsetVec;
	}
}

template<typename T>
inline void GridInterface<T>::getNeighborIndexes1D27(const iVector3& center, int indices[27]) const
{
	iVector3 indexes3D[27];
	getNeighborIndexes3D27(center, indexes3D);
	for (int i = 0; i < 27; i++)
		indices[i] = index3DTo1D(indexes3D[i]);
}

template<typename T>
inline void GridInterface<T>::getNeighborIndexes3D27(const iVector3& center, iVector3 indices[27]) const
{
	for (int i = 0; i < 27; i++)
	{
		const int* offset = neighborGridsOffset[i];
		iVector3 offsetVec(offset[0], offset[1], offset[2]);
		indices[i] = center + offsetVec;
	}
}

template<typename T>
inline void GridInterface<T>::printDatasToFile(const std::string& fileName, int numElePerLine) const
{
	std::ofstream file(fileName);
	for (int i = 0; i < (int)mVerticeDatas.size(); i++)
	{
		file << mVerticeDatas[i] << " ";
		if ((i + 1) % numElePerLine == 0)
			file << endl;
	}
	file.close();
}
