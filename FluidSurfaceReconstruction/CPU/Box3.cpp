#include "Box3.h"

void fBox3::getVertices(fVector3 vertices[8]) const
{
	//bottom facet:
	//3 2
	//0 1
	vertices[0] = fVector3(mMin.x, mMin.y, mMax.z);
	vertices[1] = fVector3(mMax.x, mMin.y, mMax.z);
	vertices[2] = fVector3(mMax.x, mMin.y, mMin.z);
	vertices[3] = fVector3(mMin.x, mMin.y, mMin.z);
	//top facet:
	//7 6
	//4 5
	vertices[4] = fVector3(mMin.x, mMax.y, mMax.z);
	vertices[5] = fVector3(mMax.x, mMax.y, mMax.z);
	vertices[6] = fVector3(mMax.x, mMax.y, mMin.z);
	vertices[7] = fVector3(mMin.x, mMax.y, mMin.z);
}

void fBox3::getCenter(fVector3 & center, Surface surface) const
{
	const fVector3& size = getSize();
	switch (surface)
	{
	case Left:
		center.setXYZ(mMin.x, mMin.y + size.y*0.5f, mMin.z + size.z*0.5f);
		break;
	case Right:
		center.setXYZ(mMax.x, mMin.y + size.y*0.5f, mMin.z + size.z*0.5f);
		break;
	case Bottom:
		center.setXYZ(mMin.x + size.x*0.5f, mMin.y, mMin.z + size.z*0.5f);
		break;
	case Top:
		center.setXYZ(mMin.x + size.x*0.5f, mMin.y, mMin.z + size.z*0.5f);
		break;
	case Back:
		center.setXYZ(mMin.x + size.x*0.5f, mMin.y + size.y*0.5f, mMin.z);
		break;
	case Front:
		center.setXYZ(mMin.x + size.x*0.5f, mMin.y + size.y*0.5f, mMax.z);
		break;
	default:
		break;
	}
}

void fBox3::getNormal(fVector3 & normal, Surface surface, bool isInside) const
{
	if (isInside)
		normal = cubeFacetsNormalsInside[surface];
	else
		normal = cubeFacetsNormalsOutside[surface];
}

void fBox3::getVertices(fVector3 vertices[4], Surface surface, bool isInside) const
{
	fVector3 allVertices[8];
	getVertices(allVertices);
	int index[4];
	if (isInside)
	{
		switch (surface)
		{
		case Left:
			index[0] = 0; index[1] = 3; index[2] = 7; index[3] = 4;
			break;
		case Right:
			index[0] = 2; index[1] = 1; index[2] = 5; index[3] = 6;
			break;
		case Bottom:
			index[0] = 1; index[1] = 0; index[2] = 3; index[3] = 2;
			break;
		case Top:
			index[0] = 5; index[1] = 4; index[2] = 7; index[3] = 6;
			break;
		case Back:
			index[0] = 3; index[1] = 2; index[2] = 6; index[3] = 7;
			break;
		case Front:
			index[0] = 1; index[1] = 0; index[2] = 4; index[3] = 5;
			break;
		default:
			break;
		}
	}
	else
	{
		switch (surface)
		{
		case Left:
			index[0] = 3; index[1] = 0; index[2] = 4; index[3] = 7;
			break;
		case Right:
			index[0] = 1; index[1] = 2; index[2] = 6; index[3] = 5;
			break;
		case Bottom:
			index[0] = 0; index[1] = 1; index[2] = 2; index[3] = 3;
			break;
		case Top:
			index[0] = 4; index[1] = 5; index[2] = 6; index[3] = 7;
			break;
		case Back:
			index[0] = 2; index[1] = 3; index[2] = 7; index[3] = 6;
			break;
		case Front:
			index[0] = 0; index[1] = 1; index[2] = 5; index[3] = 4;
			break;
		default:
			break;
		}
	}
	for (int i = 0; i < 4; i++)
		vertices[i] = allVertices[index[i]];
}

void fBox3::getTexCoords(fVector2 texCoords[4], Surface surface, float cellSize) const
{
	const fVector3& size = getSize();
	switch (surface)
	{
	case Left:
		texCoords[0] = fVector2(0.f, 0.f); texCoords[1] = fVector2(size.z, 0.f); texCoords[2] = fVector2(size.z, size.y); texCoords[3] = fVector2(0.f, size.y);
		break;
	case Right:
		texCoords[0] = fVector2(0.f, 0.f); texCoords[1] = fVector2(size.z, 0.f); texCoords[2] = fVector2(size.z, size.y); texCoords[3] = fVector2(0.f, size.y);
		break;
	case Bottom:
		texCoords[0] = fVector2(0.f, 0.f); texCoords[1] = fVector2(size.x, 0.f); texCoords[2] = fVector2(size.x, size.z); texCoords[3] = fVector2(0.f, size.z);
		break;
	case Top:
		texCoords[0] = fVector2(0.f, 0.f); texCoords[1] = fVector2(size.x, 0.f); texCoords[2] = fVector2(size.x, size.z); texCoords[3] = fVector2(0.f, size.z);
		break;
	case Back:
		texCoords[0] = fVector2(0.f, 0.f); texCoords[1] = fVector2(size.x, 0.f); texCoords[2] = fVector2(size.x, size.y); texCoords[3] = fVector2(0.f, size.y);
		break;
	case Front:
		texCoords[0] = fVector2(0.f, 0.f); texCoords[1] = fVector2(size.x, 0.f); texCoords[2] = fVector2(size.x, size.y); texCoords[3] = fVector2(0.f, size.y);
		break;
	default:
		break;
	}
	for (int i = 0; i < 4; i++)
		texCoords[i] /= cellSize;
}
