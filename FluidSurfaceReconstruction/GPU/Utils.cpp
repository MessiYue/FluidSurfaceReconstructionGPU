#include "Utils.h"

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

std::vector<SimpleParticle> Utils::readParticlesFromXYZ(
	const std::string & path,
	GridInfo * spatialGridInfo,
	float &particleRadius)
{
	std::ifstream file(path.c_str());
	std::vector<SimpleParticle> result;

	if (file)
	{
		std::cout << "Reading " << path << "...\n";
		std::string line;

		//! min point of bounding box.
		std::getline(file, line);
		std::stringstream ss1;
		ss1 << line;
		ss1 >> spatialGridInfo->minPos.x;
		ss1 >> spatialGridInfo->minPos.y;
		ss1 >> spatialGridInfo->minPos.z;

		//! max point of bounding box.
		std::getline(file, line);
		std::stringstream ss2;
		ss2 << line;
		ss2 >> spatialGridInfo->maxPos.x;
		ss2 >> spatialGridInfo->maxPos.y;
		ss2 >> spatialGridInfo->maxPos.z;

		//! kernel radius.
		std::getline(file, line);
		std::stringstream ss3;
		ss3 << line;
		ss3 >> spatialGridInfo->cellSize;

		//! particle radius.
		std::getline(file, line);
		std::stringstream ss4;
		ss4 << line;
		ss4 >> particleRadius;

		//! calculation of resoluation.
		spatialGridInfo->resolution.x = (spatialGridInfo->maxPos.x - spatialGridInfo->minPos.x)
			/ spatialGridInfo->cellSize + 1;
		spatialGridInfo->resolution.y = (spatialGridInfo->maxPos.y - spatialGridInfo->minPos.y)
			/ spatialGridInfo->cellSize + 1;
		spatialGridInfo->resolution.z = (spatialGridInfo->maxPos.z - spatialGridInfo->minPos.z)
			/ spatialGridInfo->cellSize + 1;

		//! particles' positions.
		while (std::getline(file, line))
		{
			std::stringstream str;
			str << line;
			SimpleParticle tmp;
			str >> tmp.pos.x;
			str >> tmp.pos.y;
			str >> tmp.pos.z;
			result.push_back(tmp);
		}

	}
	else
		std::cout << "Failed to read the file:" << path << std::endl;

	file.close();

	std::cout << "Finish reading " << path << "...\n";

	return result;
}

void Utils::writeTrianglesToObj(
	const std::vector<Triangle>& triangles,
	const std::string & path)
{
	//! generation of indices.
	std::vector<size_t> indices;

	std::cout << "Write to " << path << "...\n";

	std::ofstream file(path.c_str());
	if (file)
	{
		size_t count = 0;

		//! positions.
		for (const auto &elem : triangles)
		{
			file << "v " << elem.vertices[0].x << " " << elem.vertices[0].y << " " << elem.vertices[0].z << std::endl;
			indices.push_back(count++);
			file << "v " << elem.vertices[1].x << " " << elem.vertices[1].y << " " << elem.vertices[1].z << std::endl;
			indices.push_back(count++);
			file << "v " << elem.vertices[2].x << " " << elem.vertices[2].y << " " << elem.vertices[2].z << std::endl;
			indices.push_back(count++);
		}
		//! normals.
		for (const auto &elem : triangles)
		{
			file << "vn " << elem.normals[0].x << " " << elem.normals[0].y << " " << elem.normals[0].z << std::endl;
			file << "vn " << elem.normals[1].x << " " << elem.normals[1].y << " " << elem.normals[1].z << std::endl;
			file << "vn " << elem.normals[2].x << " " << elem.normals[2].y << " " << elem.normals[2].z << std::endl;
		}

		//! faces.
		//for (size_t i = 1; i <= triangles.size() * 3; i += 3)
		//{
		//	file << "f ";
		//	file << (i + 0) << "//" << (i + 0) << " ";
		//	file << (i + 1) << "//" << (i + 1) << " ";
		//	file << (i + 2) << "//" << (i + 2) << " ";
		//	file << std::endl;
		//}
		for (size_t i = 0; i < indices.size(); i += 3)
		{
			file << "f ";
			file << indices[i + 0] + 1 << "//" << indices[i + 0] + 1 << " ";
			file << indices[i + 1] + 1 << "//" << indices[i + 1] + 1 << " ";
			file << indices[i + 2] + 1 << "//" << indices[i + 2] + 1 << " ";
			file << std::endl;
		}

		file.close();
	}
	else
		std::cerr << "Failed to save " << path << std::endl;

}
