#include "Utils.h"

#include <string>
#include <sstream>
#include <fstream>
#include <memory>

#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/core/point_cloud.h>
#include <easy3d/viewer/drawable.h>

using namespace easy3d;

Utils::Configuration Utils::loadDataFromVisFile(const std::string &dir, const std::string &name)
{
	std::string path = dir + name;
	std::ifstream file(path.c_str());
	Configuration result;
	if (file)
	{
		std::string line;

		//! spatial hashing grid bounding box.

		std::getline(file, line);
		parseVector3DFFromStr(line, result.spatialMin.x, result.spatialMin.y, result.spatialMin.z);
		//std::cout << result.spatialMin.x << " " << result.spatialMin.y << " " << result.spatialMin.z << std::endl;

		std::getline(file, line);
		parseVector3DFFromStr(line, result.spatialMax.x, result.spatialMax.y, result.spatialMax.z);
		//std::cout << result.spatialMax.x << " " << result.spatialMax.y << " " << result.spatialMax.z << std::endl;

		std::getline(file, line);
		parseVector3DIFromStr(line, result.spatialRes.x, result.spatialRes.y, result.spatialRes.z);
		//std::cout << result.spatialRes.x << " " << result.spatialRes.y << " " << result.spatialRes.z << std::endl;

		std::getline(file, line);
		parseFloatFromStr(line, result.spatialCellSize);
		//std::cout << result.spatialCellSize << std::endl;

		//! scalar field grid bounding box.
		std::getline(file, line);
		parseVector3DFFromStr(line, result.scalarMin.x, result.scalarMin.y, result.scalarMin.z);
		//std::cout << result.scalarMin.x << " " << result.scalarMin.y << " " << result.scalarMin.z << std::endl;

		std::getline(file, line);
		parseVector3DFFromStr(line, result.scalarMax.x, result.scalarMax.y, result.scalarMax.z);
		//std::cout << result.scalarMax.x << " " << result.scalarMax.y << " " << result.scalarMax.z << std::endl;

		std::getline(file, line);
		parseVector3DIFromStr(line, result.scalarRes.x, result.scalarRes.y, result.scalarRes.z);
		//std::cout << result.scalarRes.x << " " << result.scalarRes.y << " " << result.scalarRes.z << std::endl;

		std::getline(file, line);
		parseFloatFromStr(line, result.scalarCellSize);
		//std::cout << result.scalarCellSize << std::endl;

		//! particles .xyz file.
		std::getline(file, line);
		result.particlesFile = dir + line;
		//std::cout << result.particlesFile << std::endl;

		//! flag array.
		unsigned int flagArraySize;
		std::getline(file, line);
		parseUIntFromStr(line, flagArraySize);
		//std::cout << flagArraySize << std::endl;
		result.flagArray.resize(flagArraySize);
		if (flagArraySize > 0)
		{
			std::getline(file, line);
			parseUIntArrayFromStr(line, result.flagArray);
			//for (size_t i = 0;i < 100;++i)
			//	std::cout << result.flagArray[i] << " ";
			//std::cout << std::endl;
		}

		//! smoothed particles.
		unsigned int numSmoothedParticles;
		std::getline(file, line);
		parseUIntFromStr(line, numSmoothedParticles);
		//std::cout << numSmoothedParticles << std::endl;
		result.smoothedParticles.resize(numSmoothedParticles);
		if (numSmoothedParticles > 0)
		{
			for (size_t i = 0; i < numSmoothedParticles; ++i)
			{
				std::getline(file, line);
				parseVector3DFFromStr(line, result.smoothedParticles[i].x, result.smoothedParticles[i].y,
					result.smoothedParticles[i].z);
				//std::cout << result.smoothedParticles[i].x << " " << result.smoothedParticles[i].y
				//	<< " " << result.smoothedParticles[i].z << std::endl;
			}
		}

		//£¡surface vertices.
		unsigned int numSurfaceVertices;
		std::getline(file, line);
		parseUIntFromStr(line, numSurfaceVertices);
		//std::cout << numSurfaceVertices << std::endl;
		result.surfaceVerticesIndices.resize(numSurfaceVertices);
		if (numSurfaceVertices > 0)
		{
			std::getline(file, line);
			parseUIntArrayFromStr(line, result.surfaceVerticesIndices);
			//for (size_t i = 0;i < 100;++i)
			//	std::cout << result.surfaceVerticesIndices[i] << " ";
			//std::cout << std::endl;
		}

		//! surface particles.
		unsigned int numSurfaceParticles;
		std::getline(file, line);
		parseUIntFromStr(line, numSurfaceParticles);
		result.surfaceParticles.resize(numSurfaceParticles);
		if (numSurfaceParticles > 0)
		{
			std::getline(file, line);
			parseUIntArrayFromStr(line, result.surfaceParticles);
		}

		//! involve particles.
		unsigned int numInvolveParticles;
		std::getline(file, line);
		parseUIntFromStr(line, numInvolveParticles);
		//std::cout << numSurfaceParticles << std::endl;
		result.involveParticles.resize(numInvolveParticles);
		if (numInvolveParticles > 0)
		{
			std::getline(file, line);
			parseUIntArrayFromStr(line, result.involveParticles);
			//for (size_t i = 0;i < 100;++i)
			//	std::cout << result.surfaceParticles[i] << " ";
			//std::cout << std::endl;
		}

		//! valid surface cubes.
		unsigned int numValidSurfaceCubes;
		std::getline(file, line);
		parseUIntFromStr(line, numValidSurfaceCubes);
		//std::cout << numValidSurfaceCubes << std::endl;
		result.validSurfaceCubes.resize(numValidSurfaceCubes);
		if (numValidSurfaceCubes > 0)
		{
			std::getline(file, line);
			parseUIntArrayFromStr(line, result.validSurfaceCubes);
			//for (size_t i = 0;i < 100;++i)
			//	std::cout << result.validSurfaceCubes[i] << " ";
			//std::cout << std::endl;
		}

		//! surface mesh file path.
		std::getline(file, line);
		result.surfaceMeshFile = dir + line;
		//std::cout << result.surfaceMeshFile << std::endl;

		//! neighbourhood extent radius.
		std::getline(file, line);
		parseFloatFromStr(line, result.neighbourhoodExtent);
		//std::cout << result.neighbourhoodExtent << std::endl;
	}
	else
		std::cerr << "Failed to read " << path << std::endl;
	
	return result;
}

void Utils::loadVisualizationScene(ViewerImGui * viewer, Configuration & config)
{
	//!------------------------------------- surface mesh ----------------------------------------------.
	SurfaceMesh *surfaceMesh = SurfaceMeshIO::load(config.surfaceMeshFile);
	ViewerImGui::surfaceMesh = surfaceMesh;
	TrianglesDrawable* surface_drawable = surfaceMesh->add_triangles_drawable("surface");
	//LinesDrawable *surface_frame_drawable = surfaceMesh->add_lines_drawable("surface_wireframe");
	// The "point" property
	auto vertices = surfaceMesh->get_vertex_property<vec3>("v:point");
	// All the XYZ coordinates
	const auto& points = vertices.vector();
	// Upload the vertex positions to the GPU.
	surface_drawable->update_vertex_buffer(points);
	//surface_frame_drawable->update_vertex_buffer(points);
	// computer vertex normals for each vertex
	surfaceMesh->update_vertex_normals();
	// The "normal" property
	auto normals = surfaceMesh->get_vertex_property<vec3>("v:normal");
	// Upload the vertex positions to the GPU.
	surface_drawable->update_normal_buffer(normals.vector());
	//surface_frame_drawable->update_normal_buffer(normals.vector());
	// Now the vertex indices for all the triangles.
	std::vector<unsigned int> indices;
	std::size_t non_triangles = 0;
	for (auto f : surfaceMesh->faces()) {
		std::vector<unsigned int> vts;
		for (auto v : surfaceMesh->vertices(f))
			vts.push_back(v.idx());
		if (vts.size() == 3)
			indices.insert(indices.end(), vts.begin(), vts.end());
		else
			++non_triangles;
	}
	surface_drawable->update_index_buffer(indices);
	//surface_frame_drawable->update_index_buffer(indices);
	surface_drawable->set_default_color(vec3(0.4f, 0.8f, 0.8f)); 
	//surface_frame_drawable->set_default_color(vec3(0.4f, 0.8f, 0.8f));
	surface_drawable->set_visible(false);
	//surface_frame_drawable->set_visible(false);

	//!----------------------------------------------- particles -------------------------------------------.
	
	//£¡original particles.
	std::vector<vec3> particlesPos = readParticlesFromXYZ(config.particlesFile, config);
	PointCloud *particles = new PointCloud;
	ViewerImGui::particles = particles;
	for (auto &elem : particlesPos)
		particles->add_vertex(elem);
	PointsDrawable* particlesDrawable = particles->add_points_drawable("particles");
	auto particlePoints = particles->get_vertex_property<vec3>("v:point");
	particlesDrawable->update_vertex_buffer(particlePoints.vector());
	particlesDrawable->set_default_color(vec3(0.0f, 0.0f, 0.9f));
	particlesDrawable->set_per_vertex_color(true);
	particlesDrawable->set_point_size(config.particleRadius);
	particlesDrawable->set_lighting(true);

	//! smoothed particles.
	PointCloud *smoothedParticles = new PointCloud;
	ViewerImGui::smoothedParticles = smoothedParticles;
	for (auto &elem : config.smoothedParticles)
		smoothedParticles->add_vertex(elem);
	PointsDrawable *smoothedParDrawable = smoothedParticles->add_points_drawable("smoothedParticles");
	auto smoothedParticlesPoints = smoothedParticles->get_vertex_property<vec3>("v:point");
	smoothedParDrawable->update_vertex_buffer(smoothedParticlesPoints.vector());
	smoothedParDrawable->set_default_color(vec3(0.9f, 0.0f, 0.0f));
	smoothedParDrawable->set_per_vertex_color(true);
	smoothedParDrawable->set_point_size(config.particleRadius);
	smoothedParDrawable->set_lighting(true);
	smoothedParDrawable->set_visible(false);

	//! surface particles.
	PointCloud *surfaceParticles = new PointCloud;
	ViewerImGui::surfaceParticles = surfaceParticles;
	for (size_t i = 0; i < config.surfaceParticles.size(); ++i)
		surfaceParticles->add_vertex(config.smoothedParticles[config.surfaceParticles[i]]);
	PointsDrawable *surfaceParDrawable = surfaceParticles->add_points_drawable("surfaceParticles");
	auto surfaceParticlesPoints = surfaceParticles->get_vertex_property<vec3>("v:point");
	surfaceParDrawable->update_vertex_buffer(surfaceParticlesPoints.vector());
	surfaceParDrawable->set_default_color(vec3(0.6, 0.6, 0.0));
	surfaceParDrawable->set_per_vertex_color(true);
	surfaceParDrawable->set_point_size(config.particleRadius);
	surfaceParDrawable->set_lighting(true);
	surfaceParDrawable->set_visible(false);

	//!------------------------------------ spatial hashing grid ---------------------------------------------.
	
	//! flag array.
	LinesDrawable *flagArrayDrawable = surfaceMesh->add_lines_drawable("spatialFlagGrid");
	flagGridForVisualization(flagArrayDrawable, config.spatialMin, config.spatialMax,
		config.spatialRes, config.spatialCellSize, config.flagArray);
	flagArrayDrawable->set_default_color(vec3(0.8f, 0.0f, 0.0f));
	flagArrayDrawable->set_visible(false);

	//! surface vertices grid.
	LinesDrawable *surfaceGridDrawable = surfaceMesh->add_lines_drawable("spatialSurfaceGrid");
	surfaceSpatialGridForVisualization(surfaceGridDrawable, config.spatialMin, config.spatialMax,
		config.spatialRes, config.spatialCellSize, config.flagArray);
	surfaceGridDrawable->set_default_color(vec3(0.0f, 0.0f, 0.8f));
	surfaceGridDrawable->set_visible(false);

	//!------------------------------------- scalar field grid ---------------------------------------------.
	LinesDrawable *surfaceCubesDrawable = surfaceMesh->add_lines_drawable("scalarSurfaceGrid");
	surfaceScalarGridForVisualization(surfaceCubesDrawable, config.scalarMin,
		config.scalarMax, config.scalarRes, config.scalarCellSize, config.surfaceVerticesIndices);
	surfaceCubesDrawable->set_default_color(vec3(0.0f, 0.8f, 0.0f));
	surfaceCubesDrawable->set_visible(false);

	LinesDrawable *validCubesDrawable = surfaceMesh->add_lines_drawable("scalarValidGrid");
	validScalarGridForVisualization(validCubesDrawable, config.scalarMin,
		config.scalarMax, config.scalarRes, config.scalarCellSize, config.validSurfaceCubes,
		config.surfaceVerticesIndices);
	validCubesDrawable->set_default_color(vec3(0.8f, 0.8f, 0.0f));
	validCubesDrawable->set_visible(false);

	viewer->add_model(particles);
	viewer->add_model(smoothedParticles);
	viewer->add_model(surfaceParticles);
	viewer->add_model(surfaceMesh);
}

unsigned int Utils::index3DTo1D(easy3d::ivec3 index3, easy3d::ivec3 res)
{
	return index3.z*res.x*res.y + index3.y*res.x + index3.x;
}

ivec3 Utils::index1DTo3D(unsigned int index1, easy3d::ivec3 res)
{
	int z = index1 / (res.x*res.y);
	int m = index1 % (res.x*res.y);
	int y = m / res.x;
	int x = m % res.x;
	return ivec3(x, y, z);
}

void Utils::parseIntFromStr(const std::string & str, int & target)
{
	std::stringstream ss;
	ss << str;
	ss >> target;
}

void Utils::parseUIntFromStr(const std::string & str, unsigned int & target)
{
	std::stringstream ss;
	ss << str;
	ss >> target;
}

void Utils::parseFloatFromStr(const std::string & str, float & target)
{
	std::stringstream ss;
	ss << str;
	ss >> target;
}

void Utils::parseUIntArrayFromStr(const std::string & str, std::vector<unsigned int>& arrayt)
{
	std::stringstream ss;
	ss << str;
	for (size_t i = 0; i < arrayt.size(); ++i)
	{
		ss >> arrayt[i];
	}
}

void Utils::parseVector3DFFromStr(const std::string & str, float & v1, float & v2, float & v3)
{
	std::stringstream ss;
	ss << str;
	ss >> v1;
	ss >> v2;
	ss >> v3;
}

void Utils::parseVector3DIFromStr(const std::string & str, int &v1, int &v2, int &v3)
{
	std::stringstream ss;
	ss << str;
	ss >> v1;
	ss >> v2;
	ss >> v3;
}

std::vector<easy3d::vec3> Utils::readParticlesFromXYZ(const std::string & path, Configuration &config)
{
	std::vector<easy3d::vec3> result;
	std::ifstream file(path.c_str());
	if (file)
	{
		std::cout << "Reading " << path << "...\n";
		std::string line;

		//! min point of bounding box.
		std::getline(file, line);

		//! max point of bounding box.
		std::getline(file, line);

		//! kernel radius.
		std::getline(file, line);

		//! particle radius.
		std::getline(file, line);
		std::stringstream ss;
		ss << line;
		ss >> config.particleRadius;

		//! particle mass.
		std::getline(file, line);

		//! particles' positions and densities.
		while (std::getline(file, line))
		{
			std::stringstream str;
			str << line;
			easy3d::vec3 tmp;
			str >> tmp.x;
			str >> tmp.y;
			str >> tmp.z;
			result.push_back(tmp);
		}

		file.close();
		std::cout << "Finish reading " << path << ".\n";
	}
	else 
		std::cout << "Failed to read the file:" << path << std::endl;
	return result;
}

void Utils::establishGridForVisualization(easy3d::LinesDrawable * drawable, easy3d::vec3 min, easy3d::vec3 max,
	easy3d::ivec3 res, float cellsize)
{
	std::vector<vec3> vertices;
	for (size_t k = 0; k < res.z; ++k)
	{
		for (size_t j = 0; j < res.y; ++j)
		{
			for (size_t i = 0; i < res.x; ++i)
			{
				vec3 corner[8];
				corner[0] = min + vec3(i * cellsize, j * cellsize, k * cellsize);
				corner[1] = corner[0] + vec3(cellsize, 0, 0);
				corner[2] = corner[0] + vec3(cellsize, 0, cellsize);
				corner[3] = corner[0] + vec3(0, 0, cellsize);
				corner[4] = corner[0] + vec3(0, cellsize, 0);
				corner[5] = corner[0] + vec3(cellsize, cellsize, 0);
				corner[6] = corner[0] + vec3(cellsize, cellsize, cellsize);
				corner[7] = corner[0] + vec3(0, cellsize, cellsize);
				vertices.push_back(corner[0]); vertices.push_back(corner[1]);
				vertices.push_back(corner[0]); vertices.push_back(corner[3]);
				vertices.push_back(corner[0]); vertices.push_back(corner[4]);
				if (i == res.x - 1)
				{
					vertices.push_back(corner[1]); vertices.push_back(corner[5]);
					vertices.push_back(corner[1]); vertices.push_back(corner[2]);
				}
				if (j == res.y - 1)
				{
					vertices.push_back(corner[4]); vertices.push_back(corner[5]);
					vertices.push_back(corner[4]); vertices.push_back(corner[7]);
				}
				if (k == res.z - 1)
				{
					vertices.push_back(corner[3]); vertices.push_back(corner[7]);
					vertices.push_back(corner[3]); vertices.push_back(corner[2]);
				}
				if (i == res.x - 1 && j == res.y - 1)
				{
					vertices.push_back(corner[5]); vertices.push_back(corner[6]);
				}
				if (i == res.x - 1 && k == res.z - 1)
				{
					vertices.push_back(corner[2]); vertices.push_back(corner[6]);
				}
				if (j == res.y - 1 && k == res.z - 1)
				{
					vertices.push_back(corner[6]); vertices.push_back(corner[7]);
				}
			}
		}
	}
	drawable->update_vertex_buffer(vertices);
}

void Utils::flagGridForVisualization(easy3d::LinesDrawable * drawable, easy3d::vec3 min,
	easy3d::vec3 max, easy3d::ivec3 res, float cellsize, std::vector<unsigned int> &flagIndex)
{
	std::vector<vec3> vertices;
	for (size_t index = 0; index < flagIndex.size(); ++index)
	{
		ivec3 index3D = index1DTo3D(flagIndex[index], res);
		vec3 corner[8];
		corner[0] = min + vec3(index3D.x * cellsize, index3D.y * cellsize, index3D.z * cellsize);
		corner[1] = corner[0] + vec3(cellsize, 0, 0);
		corner[2] = corner[0] + vec3(cellsize, 0, cellsize);
		corner[3] = corner[0] + vec3(0, 0, cellsize);
		corner[4] = corner[0] + vec3(0, cellsize, 0);
		corner[5] = corner[0] + vec3(cellsize, cellsize, 0);
		corner[6] = corner[0] + vec3(cellsize, cellsize, cellsize);
		corner[7] = corner[0] + vec3(0, cellsize, cellsize);
		vertices.push_back(corner[0]); vertices.push_back(corner[1]);
		vertices.push_back(corner[1]); vertices.push_back(corner[2]);
		vertices.push_back(corner[2]); vertices.push_back(corner[3]);
		vertices.push_back(corner[3]); vertices.push_back(corner[0]);

		vertices.push_back(corner[4]); vertices.push_back(corner[5]);
		vertices.push_back(corner[5]); vertices.push_back(corner[6]);
		vertices.push_back(corner[6]); vertices.push_back(corner[7]);
		vertices.push_back(corner[7]); vertices.push_back(corner[4]);

		vertices.push_back(corner[0]); vertices.push_back(corner[4]);
		vertices.push_back(corner[1]); vertices.push_back(corner[5]);
		vertices.push_back(corner[2]); vertices.push_back(corner[6]);
		vertices.push_back(corner[3]); vertices.push_back(corner[7]);
	}
	drawable->update_vertex_buffer(vertices);

}

void Utils::surfaceSpatialGridForVisualization(easy3d::LinesDrawable * drawable, easy3d::vec3 min,
	easy3d::vec3 max, easy3d::ivec3 res, float cellsize, std::vector<unsigned int>& flagIndex)
{
	std::vector<char> flagGrid;
	flagGrid.resize(res.x * res.y * res.z, 0);
	for (size_t i = 0; i < flagIndex.size(); ++i)
	{
		size_t index = flagIndex[i];
		flagGrid[index] = 1;
	}

	std::vector<vec3> vertices;
	for (size_t k = 0; k < res.z - 1; ++k)
	{
		for (size_t j = 0; j < res.y - 1; ++j)
		{
			for (size_t i = 0; i < res.x - 1; ++i)
			{
				ivec3 index3D(i, j, k);
				unsigned int index1D = index3DTo1D(index3D, res);
				unsigned int corners[8];
				corners[0] = index1D;
				corners[1] = index3DTo1D(index3D + ivec3(1, 0, 0), res);
				corners[2] = index3DTo1D(index3D + ivec3(1, 0, 1), res);
				corners[3] = index3DTo1D(index3D + ivec3(0, 0, 1), res);
				corners[4] = index3DTo1D(index3D + ivec3(0, 1, 0), res);
				corners[5] = index3DTo1D(index3D + ivec3(1, 1, 0), res);
				corners[6] = index3DTo1D(index3D + ivec3(1, 1, 1), res);
				corners[7] = index3DTo1D(index3D + ivec3(0, 1, 1), res);
				if (flagGrid[corners[0]] == 1 && flagGrid[corners[1]] == 1 && flagGrid[corners[2]] == 1 &&
					flagGrid[corners[3]] == 1 && flagGrid[corners[4]] == 1 && flagGrid[corners[5]] == 1 &&
					flagGrid[corners[6]] == 1 && flagGrid[corners[7]] == 1)
					continue;

				if (flagGrid[corners[0]] == 0 && flagGrid[corners[1]] == 0 && flagGrid[corners[2]] == 0 &&
					flagGrid[corners[3]] == 0 && flagGrid[corners[4]] == 0 && flagGrid[corners[5]] == 0 &&
					flagGrid[corners[6]] == 0 && flagGrid[corners[7]] == 0)
					continue;

				vec3 cornerPos[8];
				cornerPos[0] = min + vec3(index3D.x * cellsize, index3D.y * cellsize, index3D.z * cellsize);
				cornerPos[1] = cornerPos[0] + vec3(cellsize, 0, 0);
				cornerPos[2] = cornerPos[0] + vec3(cellsize, 0, cellsize);
				cornerPos[3] = cornerPos[0] + vec3(0, 0, cellsize);
				cornerPos[4] = cornerPos[0] + vec3(0, cellsize, 0);
				cornerPos[5] = cornerPos[0] + vec3(cellsize, cellsize, 0);
				cornerPos[6] = cornerPos[0] + vec3(cellsize, cellsize, cellsize);
				cornerPos[7] = cornerPos[0] + vec3(0, cellsize, cellsize);
				vertices.push_back(cornerPos[0]); vertices.push_back(cornerPos[1]);
				vertices.push_back(cornerPos[1]); vertices.push_back(cornerPos[2]);
				vertices.push_back(cornerPos[2]); vertices.push_back(cornerPos[3]);
				vertices.push_back(cornerPos[3]); vertices.push_back(cornerPos[0]);

				vertices.push_back(cornerPos[4]); vertices.push_back(cornerPos[5]);
				vertices.push_back(cornerPos[5]); vertices.push_back(cornerPos[6]);
				vertices.push_back(cornerPos[6]); vertices.push_back(cornerPos[7]);
				vertices.push_back(cornerPos[7]); vertices.push_back(cornerPos[4]);

				vertices.push_back(cornerPos[0]); vertices.push_back(cornerPos[4]);
				vertices.push_back(cornerPos[1]); vertices.push_back(cornerPos[5]);
				vertices.push_back(cornerPos[2]); vertices.push_back(cornerPos[6]);
				vertices.push_back(cornerPos[3]); vertices.push_back(cornerPos[7]);
			}
		}
	}
	drawable->update_vertex_buffer(vertices);

}

void Utils::surfaceScalarGridForVisualization(easy3d::LinesDrawable * drawable, easy3d::vec3 min, 
	easy3d::vec3 max, easy3d::ivec3 res, float cellsize, std::vector<unsigned int>& indices)
{
	std::vector<vec3> vertices;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int index1D = indices[i];
		ivec3 index3D = index1DTo3D(index1D, res);
		vec3 corner[8];
		corner[0] = min + vec3(index3D.x * cellsize, index3D.y * cellsize, index3D.z * cellsize);
		corner[1] = corner[0] + vec3(cellsize, 0, 0);
		corner[2] = corner[0] + vec3(cellsize, 0, cellsize);
		corner[3] = corner[0] + vec3(0, 0, cellsize);
		corner[4] = corner[0] + vec3(0, cellsize, 0);
		corner[5] = corner[0] + vec3(cellsize, cellsize, 0);
		corner[6] = corner[0] + vec3(cellsize, cellsize, cellsize);
		corner[7] = corner[0] + vec3(0, cellsize, cellsize);
		vertices.push_back(corner[0]); vertices.push_back(corner[1]);
		vertices.push_back(corner[1]); vertices.push_back(corner[2]);
		vertices.push_back(corner[2]); vertices.push_back(corner[3]);
		vertices.push_back(corner[3]); vertices.push_back(corner[0]);

		vertices.push_back(corner[4]); vertices.push_back(corner[5]);
		vertices.push_back(corner[5]); vertices.push_back(corner[6]);
		vertices.push_back(corner[6]); vertices.push_back(corner[7]);
		vertices.push_back(corner[7]); vertices.push_back(corner[4]);

		vertices.push_back(corner[0]); vertices.push_back(corner[4]);
		vertices.push_back(corner[1]); vertices.push_back(corner[5]);
		vertices.push_back(corner[2]); vertices.push_back(corner[6]);
		vertices.push_back(corner[3]); vertices.push_back(corner[7]);
	}
	drawable->update_vertex_buffer(vertices);

}

void Utils::validScalarGridForVisualization(easy3d::LinesDrawable * drawable, easy3d::vec3 min,
	easy3d::vec3 max, easy3d::ivec3 res, float cellsize, std::vector<unsigned int>& validIndices,
	std::vector<unsigned int>& indices)
{
	std::vector<vec3> vertices;
	for (size_t i = 0; i < validIndices.size(); ++i)
	{
		unsigned int index1D = indices[validIndices[i]];
		ivec3 index3D = index1DTo3D(index1D, res);
		vec3 corner[8];
		corner[0] = min + vec3(index3D.x * cellsize, index3D.y * cellsize, index3D.z * cellsize);
		corner[1] = corner[0] + vec3(cellsize, 0, 0);
		corner[2] = corner[0] + vec3(cellsize, 0, cellsize);
		corner[3] = corner[0] + vec3(0, 0, cellsize);
		corner[4] = corner[0] + vec3(0, cellsize, 0);
		corner[5] = corner[0] + vec3(cellsize, cellsize, 0);
		corner[6] = corner[0] + vec3(cellsize, cellsize, cellsize);
		corner[7] = corner[0] + vec3(0, cellsize, cellsize);
		vertices.push_back(corner[0]); vertices.push_back(corner[1]);
		vertices.push_back(corner[1]); vertices.push_back(corner[2]);
		vertices.push_back(corner[2]); vertices.push_back(corner[3]);
		vertices.push_back(corner[3]); vertices.push_back(corner[0]);

		vertices.push_back(corner[4]); vertices.push_back(corner[5]);
		vertices.push_back(corner[5]); vertices.push_back(corner[6]);
		vertices.push_back(corner[6]); vertices.push_back(corner[7]);
		vertices.push_back(corner[7]); vertices.push_back(corner[4]);

		vertices.push_back(corner[0]); vertices.push_back(corner[4]);
		vertices.push_back(corner[1]); vertices.push_back(corner[5]);
		vertices.push_back(corner[2]); vertices.push_back(corner[6]);
		vertices.push_back(corner[3]); vertices.push_back(corner[7]);
	}
	drawable->update_vertex_buffer(vertices);
}
