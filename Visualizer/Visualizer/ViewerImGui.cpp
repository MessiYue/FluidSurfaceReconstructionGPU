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
*	Easy3D is free software; you can redistribute it and/or modify
*	it under the terms of the GNU General Public License Version 3
*	as published by the Free Software Foundation.
*
*	Easy3D is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include "ViewerImGui.h"

#include <cmath>
#include <iostream>

#include <easy3d/util/file.h>
#include <easy3d/core/point_cloud.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/viewer/camera.h>

#include <imgui/misc/fonts/imgui_fonts_droid_sans.h>
#include <imgui/imgui.h>
#include <easy3d/util/dialogs.h>
#include <imgui/impl/imgui_impl_glfw.h>
#include <imgui/impl/imgui_impl_opengl3.h>
#include <glfw/include/GLFW/glfw3.h>

#include "Utils.h"

namespace easy3d {

	ImGuiContext* ViewerImGui::context_ = nullptr;

	easy3d::SurfaceMesh *ViewerImGui::surfaceMesh = nullptr;
	easy3d::PointCloud *ViewerImGui::particles = nullptr;
	easy3d::PointCloud *ViewerImGui::smoothedParticles = nullptr;
	easy3d::PointCloud *ViewerImGui::surfaceParticles = nullptr;

	ViewerImGui::ViewerImGui(
		const std::string& title /* = "Easy3D ImGui Viewer" */,
		int samples /* = 4 */,
		int gl_major /* = 3 */,
		int gl_minor /* = 2 */,
		bool full_screen /* = false */,
		bool resizable /* = true */,
		int depth_bits /* = 24 */,
		int stencil_bits /* = 8 */)
		: Viewer(title, samples, gl_major, gl_minor, full_screen, resizable, depth_bits, stencil_bits)
		, alpha_(0.8f)
		, movable_(true)
	{
	}


	void ViewerImGui::init() {
		Viewer::init();

		if (!context_) {
			// Setup ImGui binding
			IMGUI_CHECKVERSION();

			context_ = ImGui::CreateContext();

			const char* glsl_version = "#version 150";
			ImGui_ImplGlfw_InitForOpenGL(window_, false);
			ImGui_ImplOpenGL3_Init(glsl_version);
			ImGuiIO& io = ImGui::GetIO();
			io.WantCaptureKeyboard = true;
			io.WantTextInput = true;
			io.IniFilename = nullptr;
			ImGui::StyleColorsDark();
			ImGuiStyle& style = ImGui::GetStyle();
			style.FrameRounding = 5.0f;

			// load font
			reload_font();
		}
	}


	double ViewerImGui::pixel_ratio() {
		// Computes pixel ratio for hidpi devices
		int fbo_size[2], win_size[2];
		glfwGetFramebufferSize(window_, &fbo_size[0], &fbo_size[1]);
		glfwGetWindowSize(window_, &win_size[0], &win_size[1]);
		return static_cast<double>(fbo_size[0]) / static_cast<double>(win_size[0]);
	}


	void ViewerImGui::reload_font(int font_size)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.Fonts->Clear();
		io.Fonts->AddFontFromMemoryCompressedTTF(droid_sans_compressed_data, droid_sans_compressed_size, font_size * dpi_scaling());
		io.FontGlobalScale = 1.0f / pixel_ratio();
		ImGui_ImplOpenGL3_DestroyDeviceObjects();
	}


	void ViewerImGui::post_resize(int w, int h) {
		Viewer::post_resize(w, h);
		if (context_) {
			ImGui::GetIO().DisplaySize.x = float(w);
			ImGui::GetIO().DisplaySize.y = float(h);
		}
	}


	bool ViewerImGui::callback_event_cursor_pos(double x, double y) {
		if (ImGui::GetIO().WantCaptureMouse)
			return true;
		else
			return Viewer::callback_event_cursor_pos(x, y);
	}


	bool ViewerImGui::callback_event_mouse_button(int button, int action, int modifiers) {
		if (ImGui::GetIO().WantCaptureMouse)
			return true;
		else
			return Viewer::callback_event_mouse_button(button, action, modifiers);
	}


	bool ViewerImGui::callback_event_keyboard(int key, int action, int modifiers) {
		if (ImGui::GetIO().WantCaptureKeyboard)
			return true;
		else
			return Viewer::callback_event_keyboard(key, action, modifiers);
	}


	bool ViewerImGui::callback_event_character(unsigned int codepoint) {
		if (ImGui::GetIO().WantCaptureKeyboard)
			return true;
		else
			return Viewer::callback_event_character(codepoint);
	}


	bool ViewerImGui::callback_event_scroll(double dx, double dy) {
		if (ImGui::GetIO().WantCaptureMouse)
			return true;
		else
			return Viewer::callback_event_scroll(dx, dy);
	}


	void ViewerImGui::cleanup() {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();

		ImGui::DestroyContext(context_);

		Viewer::cleanup();
	}


	void ViewerImGui::pre_draw() {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		Viewer::pre_draw();
	}


	void ViewerImGui::draw_overlay(bool* visible)
	{
		ImGui::SetNextWindowSize(ImVec2(300 * widget_scaling(), 200 * widget_scaling()), ImGuiCond_FirstUseEver);
		const float distance = 10.0f;
		static int corner = 1;
		ImVec2 window_pos = ImVec2(ImGui::GetIO().DisplaySize.x - distance, distance + 30);
		ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
		if (corner != -1)
			ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
		ImGui::SetNextWindowBgAlpha(alpha_); // Transparent background
		if (ImGui::Begin("Easy3D: Information", visible, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize/* | ImGuiWindowFlags_AlwaysAutoResize*/ | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
		{
			ImGui::Text("Info (right-click to change position)");
			ImGui::Separator();
			ImGui::Text("Frame rate: %.1f", ImGui::GetIO().Framerate);
			//ImGui::Text("GPU time (ms): %4.1f", gpu_time_);

			if (current_model()) {
				const std::string& name = "Current model: " + file::simple_name(current_model()->name());
				ImGui::Text("%s", name.c_str());
				if (dynamic_cast<PointCloud*>(current_model())) {
					PointCloud* cloud = dynamic_cast<PointCloud*>(current_model());
					ImGui::Text("Type: point cloud");
					ImGui::Text("#Vertices: %i", cloud->n_vertices());
				}
				else if (dynamic_cast<SurfaceMesh*>(current_model())) {
					SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(current_model());
					ImGui::Text("Type: surface mesh");
					ImGui::Text("#Faces: %i", mesh->n_faces());
					ImGui::Text("#Vertices: %i", mesh->n_vertices());
					ImGui::Text("#Edges: %i", mesh->n_edges());
				}
			}
			ImGui::Separator();

			int w, h;
			glfwGetWindowSize(window_, &w, &h);
			float x = ImGui::GetIO().MousePos.x;
			float y = ImGui::GetIO().MousePos.y;
			ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", camera()->position().x, camera()->position().y, camera()->position().z);
			ImGui::Text("Camera Direction: (%.2f, %.2f, %.2f)", camera()->viewDirection().x, camera()->viewDirection().y, camera()->viewDirection().z);
			if (ImGui::BeginPopupContextWindow())
			{
				if (ImGui::MenuItem("Custom", nullptr, corner == -1)) corner = -1;
				if (ImGui::MenuItem("Top-left", nullptr, corner == 0)) corner = 0;
				if (ImGui::MenuItem("Top-right", nullptr, corner == 1)) corner = 1;
				if (ImGui::MenuItem("Bottom-left", nullptr, corner == 2)) corner = 2;
				if (ImGui::MenuItem("Bottom-right", nullptr, corner == 3)) corner = 3;
				if (visible && ImGui::MenuItem("Close")) *visible = false;
				ImGui::EndPopup();
			}
		}
		ImGui::End();
	}

	void ViewerImGui::pathSplit(const std::string & path, std::string & name, std::string & directory)
	{
		// split the path string into dir and file name.
		size_t index = path.find_last_of("\\");
		directory = path.substr(0, index + 1);
		name = path.substr(index + 1, path.size() - index - 1);
	}

	void ViewerImGui::post_draw() {
		static bool show_overlay = true;
		if (show_overlay)
			draw_overlay(&show_overlay);

		//! visualization options.
		static bool show_vis_options = true;
		ImGui::Begin("Visualization", &show_vis_options, ImGuiWindowFlags_NoResize);
		ImGui::Checkbox("draw original particles", &drawParticles_);
		ImGui::Checkbox("draw smoothed particles", &drawSmoothedParticles_);
		ImGui::Checkbox("draw surface particles", &drawSurfaceParticles_);
		ImGui::Checkbox("draw non empty spatial hashing cell", &drawNonEmptySpatialGrid_);
		ImGui::Checkbox("draw surface spatial hashing cell", &drawSurfaceSpatialGrid_);
		ImGui::Checkbox("draw surface scalar field cell", &drawSurfaceScalarGrid_);
		ImGui::Checkbox("draw valid scalar field cell", &drawValidScalarGird_);
		ImGui::Checkbox("draw fluid surface mesh", &drawMeshFilling_);
		ImGui::Checkbox("draw fluid surface mesh in wireframe mode", &drawMeshTriangles_);
		ImGui::End();

		if (ViewerImGui::surfaceMesh != nullptr)
		{
			ViewerImGui::surfaceMesh->triangles_drawable("surface")->set_visible(drawMeshFilling_);
			ViewerImGui::particles->points_drawable("particles")->set_visible(drawParticles_);
			ViewerImGui::smoothedParticles->points_drawable("smoothedParticles")->set_visible(drawSmoothedParticles_);
			ViewerImGui::surfaceParticles->points_drawable("surfaceParticles")->set_visible(drawSurfaceParticles_);
			ViewerImGui::surfaceMesh->lines_drawable("spatialFlagGrid")->set_visible(drawNonEmptySpatialGrid_);
			ViewerImGui::surfaceMesh->lines_drawable("spatialSurfaceGrid")->set_visible(drawSurfaceSpatialGrid_);
			ViewerImGui::surfaceMesh->lines_drawable("scalarSurfaceGrid")->set_visible(drawSurfaceScalarGrid_);
			ViewerImGui::surfaceMesh->lines_drawable("scalarValidGrid")->set_visible(drawValidScalarGird_);
			//ViewerImGui::surfaceMesh->lines_drawable("surface_wireframe")->set_visible(drawMeshTriangles_);
		}

		static bool show_about = false;
		if (show_about) {
			ImGui::SetNextWindowPos(ImVec2(width() * 0.5f, height() * 0.5f), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
			ImGui::Begin("About Easy3D ImGui Viewer", &show_about, ImGuiWindowFlags_NoResize);
			ImGui::Text(
				"This viewer shows how to use ImGui for GUI creation and event handling"
			);
			ImGui::Separator();
			ImGui::Text(
				"\n"
				"Liangliang Nan\n"
				"liangliang.nan@gmail.com\n"
				"https://3d.bk.tudelft.nl/liangliang/\n"
			);
			ImGui::End();
		}

		static bool show_manual = false;
		if (show_manual) {
			int w, h;
			glfwGetWindowSize(window_, &w, &h);
			ImGui::SetNextWindowPos(ImVec2(w * 0.5f, h * 0.5f), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
			ImGui::Begin("Easy3D Manual", &show_manual, ImGuiWindowFlags_NoResize);
			ImGui::Text("%s", usage().c_str());
			ImGui::End();
		}

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 8));
		if (ImGui::BeginMainMenuBar())
		{
			draw_menu_file();

			draw_menu_view();

			if (ImGui::BeginMenu("Help"))
			{
				ImGui::MenuItem("Manual", nullptr, &show_manual);
				ImGui::Separator();
				ImGui::MenuItem("About", nullptr, &show_about);
				ImGui::EndMenu();
			}
			menu_height_ = ImGui::GetWindowHeight();
			ImGui::EndMainMenuBar();
		}
		ImGui::PopStyleVar();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		Viewer::post_draw();
	}


	void ViewerImGui::draw_menu_file() 
	{
		static std::string defaultDirectory = "";
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Open", "Ctrl+O"))
				open();
			// load visualization file.
			if (ImGui::MenuItem("Open vis", "Ctrl+G"))
			{
				const std::vector<std::string> filetypes = { "" };
				const std::vector<std::string>& file_names = FileDialog::open(filetypes, true, defaultDirectory);
				if (file_names.size() > 0)
				{
					std::string dir, name;
					pathSplit(file_names[0], name, dir);
					std::cout << dir << " " << name << std::endl;
					Utils::Configuration config = Utils::loadDataFromVisFile(dir, name);
					this->delete_models();
					Utils::loadVisualizationScene(this, config); 
					//this->camera()->lookAt(this->camera()->sceneCenter());
					//this->camera()->setViewDirection(vec3(-0.75, -0.35, 0.56));
				}
			}
			if (ImGui::MenuItem("Save As...", "Ctrl+S"))
				save();

			//ImGui::Separator();
			//if (ImGui::BeginMenu("Recent Files...")) {
			//	std::string file_name;
			//	std::vector<Model*>::const_reverse_iterator it = models_.rbegin();
			//	for (; it != models_.rend(); ++it) {
			//		if (ImGui::MenuItem((*it)->name().c_str())) {
			//			file_name = (*it)->name();
			//		}
			//	}
			//	if (!file_name.empty())
			//		open(file_name);
			//	ImGui::EndMenu();
			//}

			ImGui::Separator();
			if (ImGui::MenuItem("Quit", "Alt+F4"))
				glfwSetWindowShouldClose(window_, GLFW_TRUE);

			ImGui::EndMenu();
		}
	}


	void ViewerImGui::draw_menu_view() {
		if (ImGui::BeginMenu("View"))
		{
			if (ImGui::MenuItem("Snapshot", nullptr))
				snapshot();

			ImGui::Separator();
			if (ImGui::BeginMenu("Options"))
			{
				ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.50f);

				static int style_idx = 1;
				if (ImGui::Combo("Style", &style_idx, "Classic\0Dark\0Light\0")) {
					switch (style_idx) {
					case 0: ImGui::StyleColorsClassic(); break;
					case 1: ImGui::StyleColorsDark(); break;
					case 2: ImGui::StyleColorsLight(); break;
					}
				}

				ImGui::Checkbox("Panel Movable", &movable_);
				ImGui::ColorEdit3("Background Color", (float*)background_color_, ImGuiColorEditFlags_NoInputs);
				ImGui::DragFloat("Transparency", &alpha_, 0.005f, 0.0f, 1.0f, "%.1f");
				ImGui::PopItemWidth();
				ImGui::EndMenu();
			}

			ImGui::EndMenu();
		}
	}
}
