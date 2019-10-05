#include "MarchingCubes.h"

void MarchingCubes::getTriangles(const iVector3& index3D, Triangle triangles[5], int& triCount) const
{
	triCount = 0;

	// 8 corner indices.
	iVector3 indexes3D[8];
	MarchingCubesHelper::getCornerIndexes3D(index3D, indexes3D);

	// 8 corner vertices.
	GridVertex* vertices[8];
	for (int i = 0; i < 8; i++)
		vertices[i] = mScalarGrid->getVertexDataP(indexes3D[i]);

	// ��8λ���ڱ�־8���������ֵ���λ�ù�ϵ
	int vertexFlag = 0;
	for (int i = 0; i < 8; i++)
	{
		// λ�ڵ�ֵ���ϻ��ֵ���һ��
		if (vertices[i]->value <= mIsoValue)
			vertexFlag |= 1 << i;
	}

	// ��ǰcube���ж�����ڵ�ֵ��һ��
	if (vertexFlag == 0 || vertexFlag == 255)
		return;

	// ��12λ���ڱ��12�������ֵ���ཻ���
	unsigned int edgeFlag = MarchingCubesHelper::edgeFlags[vertexFlag];

	// 12�����ϵĽ���λ��
	fVector3 intersectPoss[12];
	// 12�����ϵĽ��㷨����
	fVector3 intersectNormals[12];
	for (int i = 0; i < 12; i++)
	{
		// ���Ϊi�ı����ֵ���ཻ
		if (edgeFlag & (1 << i))
		{
			int startVertex = MarchingCubesHelper::vertexIndexesOfEdge[i][0];
			int endVertex = MarchingCubesHelper::vertexIndexesOfEdge[i][1];
			float offsetPer = getOffsetFac(vertices[startVertex]->value, 
				vertices[endVertex]->value, mIsoValue);
			fVector3 startPos = vertices[startVertex]->position;
			fVector3 endPos = vertices[endVertex]->position;
			fVector3 intersectPos = getOffsetVector(startPos, endPos, offsetPer);
			intersectPoss[i] = intersectPos;

			if (!vertices[startVertex]->isNormalCal)
			{
				vertices[startVertex]->normal = mScalarGrid->calcNormal(indexes3D[startVertex]);
				vertices[startVertex]->normal.normalize();
				vertices[startVertex]->isNormalCal = true;
			}
			if (!vertices[endVertex]->isNormalCal)
			{
				vertices[endVertex]->normal = mScalarGrid->calcNormal(indexes3D[endVertex]);
				vertices[endVertex]->normal.normalize();
				vertices[endVertex]->isNormalCal = true;
			}
			fVector3 startNor = vertices[startVertex]->normal;
			fVector3 endNor = vertices[endVertex]->normal;
			intersectNormals[i] = getOffsetVector(startNor, endNor, offsetPer);
			intersectNormals[i].normalize();
		}
	}
	for (int i = 0; i < 5; i++)//ÿ��cube������5��������
	{
		if (MarchingCubesHelper::edgeIndexesOfTriangle[vertexFlag][i * 3] < 0)
			break;
		else
		{
			for (int j = 0; j < 3; j++)
			{
				int edgeIndex = MarchingCubesHelper::edgeIndexesOfTriangle[vertexFlag][i * 3 + j];
				triangles[triCount].vertices[j] = intersectPoss[edgeIndex];
				triangles[triCount].normals[j] = intersectNormals[edgeIndex];
			}
			++triCount;
		}
	}
}

float MarchingCubes::getOffsetFac(float val0, float val1, float targetVal) const
{
	float delta = val1 - val0;
	if (delta > -EPSILON_MATH && delta < EPSILON_MATH)
		return 0.5;
	return (targetVal - val0) / delta;
}

fVector3 MarchingCubes::getOffsetVector(const fVector3 & v0, const fVector3 & v1, float offsetFac) const
{
	fVector3 delta = v1 - v0;
	return v0 + delta * offsetFac;
}

fVector3 MarchingCubes::getGradient(const iVector3 & index3D) const
{
	float val0 = 0;
	float val1 = 0;
	mScalarGrid->getValue(iVector3(index3D.x - 1, index3D.y, index3D.z), val0);
	mScalarGrid->getValue(iVector3(index3D.x + 1, index3D.y, index3D.z), val1);
	float gradX = val0 - val1;

	mScalarGrid->getValue(iVector3(index3D.x, index3D.y - 1, index3D.z), val0);
	mScalarGrid->getValue(iVector3(index3D.x, index3D.y + 1, index3D.z), val1);
	float gradY = val0 - val1;

	mScalarGrid->getValue(iVector3(index3D.x, index3D.y, index3D.z - 1), val0);
	mScalarGrid->getValue(iVector3(index3D.x, index3D.y, index3D.z + 1), val1);
	float gradZ = val0 - val1;

	return fVector3(gradX, gradY, gradZ);
}
