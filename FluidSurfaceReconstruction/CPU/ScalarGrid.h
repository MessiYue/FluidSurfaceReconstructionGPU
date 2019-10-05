#pragma once

#include <vector>

#include "Box3.h"
#include "MathVector.h"
#include "GridInterface.h"

struct GridVertex 
{
	float value;
	fVector3 position;
	fVector3 normal;
	bool isNormalCal;
	bool isNearSurface;

	GridVertex() : value(0.f), isNormalCal(false), isNearSurface(false) {}

};

class ScalarGrid : public GridInterface<GridVertex>
{
public:

	typedef std::shared_ptr<ScalarGrid> ptr;

	ScalarGrid() {}
	ScalarGrid(const fBox3& gridBox, float cellSize) { init(gridBox, cellSize); updatePositions(); }
	ScalarGrid(const fBox3& gridBox, const iVector3& gridRes) { init(gridBox, gridRes); updatePositions(); }

	void reset(float val = 0.f, bool isNorCal = false);
	void resetE(float val = 0.f, bool isNorCal = false);

	float getValue(int index1D);
	bool getValue(int index1D, float& _value);

	float getValue(const iVector3& index3D);
	bool getValue(const iVector3& index3D, float& _value);

	fVector3 getNormal(int index1D) const;
	fVector3 getNormal(const iVector3& index3D) const;
	const fVector3 calcNormal(const iVector3& index3D);

	void updateNormals();
	void updatePositions();

	std::vector<GridVertex>& getVerticeData() { return mVerticeDatas; }

};

inline std::ostream& operator<<(std::ostream& os, const GridVertex& a)
{
	os << a.value;
	return os;
}
