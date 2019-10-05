#pragma once

#include "MathVector.h"

class MarchingCubesHelper
{
public:
	//! [�߱��][��������]��ֵΪ�ñߵ�����������
	static const unsigned int vertexIndexesOfEdge[12][2];

	//! ������8�����������0�Ŷ����ƫ�Ƶ�λֵ
	static const int cubeVerticesOffset[8][3];

	//! �ӳ�����ĳ������㵽�յ�ĵ�λ����
	static const float cubeEdgeDirection[12][3];

	//! [���8���������ֵ��λ�ù�ϵ��8λ��������]��ֵΪ���12�������ֵ���ཻ����������ֻ�õ�12λ��
	static const unsigned int edgeFlags[256];

	//! [���8���������ֵ��λ�ù�ϵ��8λ��������][���15�����㣨���������Σ���
	//! ���һ��Ԫ�����ڽ������]��ֵΪ�����������ζ�������cube�ıߺ�
	//! �������������������˳��Ϊ��ʱ�뷽��
	static const int edgeIndexesOfTriangle[256][16];

	//! ��Ӧcell/voxel�����Ķ�����
	static const unsigned int numVertices[256];

	//! ��ȡindex3D��������cube��8������
	static void getCornerIndexes3D(const iVector3& index3D, iVector3 indexes3D[8]);

};

