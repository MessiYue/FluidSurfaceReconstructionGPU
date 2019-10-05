#pragma once

#include <cmath>
#include "MathVector.h"

#ifndef max__
#define max__(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min__
#define min__(a,b)            (((a) < (b)) ? (a) : (b))
#endif

const float PI = 3.141592653f;

const float EPSILON_MATH = (float)1.0e-7;

inline bool isEqual(float a, float b)
{
	float c = a - b;
	return c >= -EPSILON_MATH && c <= EPSILON_MATH;
}

inline bool isZero(float a)
{
	return a > -EPSILON_MATH && a < EPSILON_MATH;
}

enum Surface { Left, Right, Bottom, Top, Back, Front, NDF };

const fVector3 cubeFacetsNormalsOutside[6] =
{
	fVector3(-1.f,0.f,0.f),
	fVector3(1.f,0.f,0.f),
	fVector3(0.f,-1.f,0.f),
	fVector3(0.f,1.f,0.f),
	fVector3(0.f,0.f,-1.f),
	fVector3(0.f,0.f,1.f)
};

const fVector3 cubeFacetsNormalsInside[6] =
{
	fVector3(1.f,0.f,0.f),
	fVector3(-1.f,0.f,0.f),
	fVector3(0.f,1.f,0.f),
	fVector3(0.f,-1.f,0.f),
	fVector3(0.f,0.f,1.f),
	fVector3(0.f,0.f,-1.f)
};

const int cubeVerticesOffset[8][3] =
{
	{0, 0, 0},{1, 0, 0},{1, 1, 0},{0, 1, 0},
	{0, 0, 1},{1, 0, 1},{1, 1, 1},{0, 1, 1}
};

const int neighborGridsOffset[27][3] =
{
	{-1,-1,-1},{0,-1,-1},{1,-1,-1},{-1,0,-1},{0,0,-1},{1,0,-1},{-1,1,-1},{0,1,-1},{1,1,-1},
	{-1,-1,0},{0,-1,0},{1,-1,0},{-1,0,0},{0,0,0},{1,0,0},{-1,1,0},{0,1,0},{1,1,0},
	{-1,-1,1},{0,-1,1},{1,-1,1},{-1,0,1},{0,0,1},{1,0,1},{-1,1,1},{0,1,1},{1,1,1}
};

struct Triangle 
{
	fVector3 vertices[3];
	fVector3 normals[3];
};

