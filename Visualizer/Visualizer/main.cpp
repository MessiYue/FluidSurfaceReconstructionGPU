/*
*	Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
*	https://3d.bk.tudelft.nl/liangliang/
*
*	This file is part of Easy3D. If it is useful in your research/work,
*   I would be grateful if you show your appreciation by citing it:
*   ------------------------------------------------------------------
*           Liangliang Nan.
*           Easy3D: a lightweight, easy-to-use, and efficient C++
*           library for processing and rendering 3D data. 2018.
*   ------------------------------------------------------------------
*
*	EasyGUI is free software; you can redistribute it and/or modify
*	it under the terms of the GNU General Public License Version 3
*	as published by the Free Software Foundation.
*
*	EasyGUI is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>

#include "ViewerImGui.h"
#include <easy3d/viewer/setting.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/viewer/drawable.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/viewer/camera.h>

#include "Utils.h"

using namespace easy3d;

int main(int argc, char** argv)
{
	try 
	{
		ViewerImGui viewer("Visualization for fluid surface reconstruction", 80, 3, 2);

		viewer.resize(1000, 800);
				
		viewer.camera()->setZNearCoefficient(0.001f);
		viewer.camera()->setZClippingCoefficient(100);
		viewer.set_background_color(vec4(0, 0, 0, 1));

		viewer.run();

	}
	catch (const std::runtime_error &e) {
		std::string error_msg = std::string("Caught a fatal error: ") + std::string(e.what());
		std::cerr << error_msg << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
