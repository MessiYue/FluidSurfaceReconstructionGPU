#pragma once

#include "MathVector.h"

class MarchingCubesHelper
{
public:
	//! [边编号][两个顶点]，值为该边的两个顶点编号
	static const unsigned int vertexIndexesOfEdge[12][2];

	//! 长方体8个顶点相对于0号顶点的偏移单位值
	static const int cubeVerticesOffset[8][3];

	//! 从长方体某条边起点到终点的单位向量
	static const float cubeEdgeDirection[12][3];

	//! [标记8个顶点与等值面位置关系的8位二进制数]，值为标记12条边与等值面相交与否的整数（只用低12位）
	static const unsigned int edgeFlags[256];

	//! [标记8个顶点与等值面位置关系的8位二进制数][最多15个顶点（三个三角形），
	//! 最后一个元素用于结束标记]，值为产生的三角形顶点所在cube的边号
	//! 三角形三个顶点的连接顺序为逆时针方向
	static const int edgeIndexesOfTriangle[256][16];

	//! 对应cell/voxel产生的顶点数
	static const unsigned int numVertices[256];

	//! 获取index3D顶点所在cube的8个顶点
	static void getCornerIndexes3D(const iVector3& index3D, iVector3 indexes3D[8]);

};

