#include "SurfaceReconstructor.h"

SurfaceReconstructor::SurfaceReconstructor()
	: MarchingCubes() {}

SurfaceReconstructor::SurfaceReconstructor(ScalarGrid::ptr scalarGrid, float isoValue)
	: MarchingCubes(scalarGrid, isoValue) {}

SurfaceReconstructor::~SurfaceReconstructor() {}

void SurfaceReconstructor::generateTriangles()
{
	mPosArray.clear();
	mNorArray.clear();
	iVector3 gridRes = mScalarGrid->getGridRes();

	for (int z = 0; z < gridRes.z; z++)
	{
		for (int y = 0; y < gridRes.y; y++)
		{
			for (int x = 0; x < gridRes.x; x++)
			{
				iVector3 index(x, y, z);
				//if ((mSfConfig->sfRecMethod == Method2014
				//	|| mSfConfig->sfRecMethod == Method2016 /*&& mSfConfig->isoSurMethod == IsoTMC01*/
				//	|| mSfConfig->sfRecMethod == Method2016New2
				//	|| mSfConfig->sfRecMethod == MethodAkinci && mSfConfig->isoSurMethod == IsoZB05)
				//	&& !mScalarGrid->getVertexDataP(index)->isNearSurface)
				//	continue;

				bool toGerTri = true;
				//{
				//	iVector3 indexes3D[8];
				//	MarchingCubesHelper::getCornerIndexes3D(index, indexes3D);
				//	for (int i = 1; i < 8; i++)
				//	{
				//		if (!mScalarGrid->getVertexDataP(indexes3D[i])->isNearSurface)
				//		{
				//			toGerTri = false;
				//			break;
				//		}
				//	}
				//}
				if (!toGerTri)
					continue;

				Triangle triangles[5];
				int triCount = 0;
				getTriangles(index, triangles, triCount);
				for (int i = 0; i < triCount; i++)
				{
					mPosArray.push_back(triangles[i].vertices[0]);
					mNorArray.push_back(triangles[i].normals[0]);

					mPosArray.push_back(triangles[i].vertices[1]);
					mNorArray.push_back(triangles[i].normals[1]);

					mPosArray.push_back(triangles[i].vertices[2]);
					mNorArray.push_back(triangles[i].normals[2]);
				}
			}//end for (int z=0;z<gridRes.z;z++)
		}//end for(int y=0;y<gridRes.y;y++)
	}//end for(int x=0;x<gridRes.x;x++)

}
