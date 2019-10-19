#pragma once

#include <vector>
#include <string>

#include <easy3d/core/surface_mesh.h>
#include <easy3d/viewer/drawable.h>

#include "ViewerImGui.h"

class Utils
{
public:

	struct Configuration
	{
	public:
		//! spatial hashing grid bounding box.
		easy3d::vec3 spatialMin;
		easy3d::vec3 spatialMax;
		easy3d::ivec3 spatialRes;
		float spatialCellSize;

		//! scalar field grid bounding box.
		easy3d::vec3 scalarMin;
		easy3d::vec3 scalarMax;
		easy3d::ivec3 scalarRes;
		float scalarCellSize;

		//! particles .xyz file.
		std::string particlesFile;

		//! flag array.
		std::vector<unsigned int> flagArray;

		//! smoothed particles.
		std::vector<easy3d::vec3> smoothedParticles;

		//£¡surface vertices.
		std::vector<unsigned int> surfaceVerticesIndices;

		//! surface particles.
		std::vector<unsigned int> surfaceParticles;

		//! involve particles.
		std::vector<unsigned int> involveParticles;
		
		//! valid surface cubes.
		std::vector<unsigned int> validSurfaceCubes;

		//! surface mesh file path.
		std::string surfaceMeshFile;

		//! neighbourhood extent radius.
		float neighbourhoodExtent;

		//! particle radius.
		float particleRadius;

	};

	static Configuration loadDataFromVisFile(const std::string &dir, const std::string &name);
	static void loadVisualizationScene(easy3d::ViewerImGui *viewer, Configuration &config);

private:
	static unsigned int index3DTo1D(easy3d::ivec3 index3, easy3d::ivec3 res);
	static easy3d::ivec3 index1DTo3D(unsigned int index1, easy3d::ivec3 res);

	static void parseIntFromStr(const std::string &str, int &target);
	static void parseUIntFromStr(const std::string &str, unsigned int &target);
	static void parseFloatFromStr(const std::string &str, float &target);
	static void parseUIntArrayFromStr(const std::string &str, std::vector<unsigned int> &arrayt);
	static void parseVector3DFFromStr(const std::string &str, float &v1, float &v2, float &v3);
	static void parseVector3DIFromStr(const std::string &str, int &v1, int &v2, int &v3);
	static std::vector<easy3d::vec3> readParticlesFromXYZ(const std::string &str, Configuration &config);

	static void establishGridForVisualization(easy3d::LinesDrawable *drawable, easy3d::vec3 min,
		easy3d::vec3 max, easy3d::ivec3 res, float cellsize);

	static void flagGridForVisualization(easy3d::LinesDrawable *drawable, easy3d::vec3 min,
		easy3d::vec3 max, easy3d::ivec3 res, float cellsize, std::vector<unsigned int> &flagIndex);

	static void surfaceSpatialGridForVisualization(easy3d::LinesDrawable *drawable, easy3d::vec3 min,
		easy3d::vec3 max, easy3d::ivec3 res, float cellsize, std::vector<unsigned int> &flagIndex);

	static void surfaceScalarGridForVisualization(easy3d::LinesDrawable *drawable, easy3d::vec3 min,
		easy3d::vec3 max, easy3d::ivec3 res, float cellsize, std::vector<unsigned int> &indices);

	static void validScalarGridForVisualization(easy3d::LinesDrawable *drawable, easy3d::vec3 min,
		easy3d::vec3 max, easy3d::ivec3 res, float cellsize, std::vector<unsigned int> &validIndices,
		std::vector<unsigned int> &indices);

};