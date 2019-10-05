#pragma once

#include "MathUtils.h"
#include "MathVector.h"

class iBox3
{
public:

	iBox3() {}
	iBox3(const iVector3& _min, const iVector3& _max) :mMin(_min), mMax(_max) {}
	iBox3(const iBox3& box) :mMin(box.getMin()), mMax(box.getMax()) {}
	iBox3(int xMin, int yMin, int zMin, int xMax, int yMax, int zMax)
		:mMin(xMin, yMin, zMin), mMax(xMax, yMax, zMax) {}

	const iVector3& getMin() const { return mMin; }
	const iVector3& getMax() const { return mMax; }
	iVector3 getSize() const { return iVector3(mMax - mMin); }

	void setMin(const iVector3& _min) { mMin = _min; }
	void setMax(const iVector3& _max) { mMax = _max; }

private:
	iVector3 mMin;
	iVector3 mMax;
};

class fBox3
{
public:
	fBox3() {}
	fBox3(const fBox3& box) :mMin(box.getMin()), mMax(box.getMax()) {}
	fBox3(const fVector3& _min, const fVector3& _max) :mMin(_min), mMax(_max) {}
	fBox3(float xMin, float yMin, float zMin, float xMax, float yMax, float zMax)
		:mMin(xMin, yMin, zMin), mMax(xMax, yMax, zMax) {}

	//bottom facet:
	//3 2
	//0 1
	//top facet:
	//7 6
	//4 5
	void getVertices(fVector3 vertices[8]) const;

	//isInside:视点是否在box里，由逆时针方向取点决定正面
	//3 2
	//0 1
	void getCenter(fVector3& center, Surface surface) const;
	void getNormal(fVector3& normal, Surface surface, bool isInside = false) const;
	void getVertices(fVector3 vertices[4], Surface surface, bool isInside = false) const;
	void getTexCoords(fVector2 texCoords[4], Surface surface, float cellSize = 1.0f) const;

	const fVector3& getMin() const { return mMin; }
	const fVector3& getMax() const { return mMax; }
	fVector3 getSize() const { return fVector3(mMax - mMin); }
	const fVector3 getCenter() const { return (mMin + mMax) / 2.f; }

	void setMin(const fVector3& _min) { mMin = _min; }
	void setMax(const fVector3& _max) { mMax = _max; }

private:
	fVector3 mMin;
	fVector3 mMax;
};


