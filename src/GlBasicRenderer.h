//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_GLBASICRENDERER_H
#define TRIANGLE_RAYCASTING_GLBASICRENDERER_H

#include "Renderer.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Camera.h"

// a class handling rendering the scene with OpenGL
class GlBasicRenderer : public Renderer
{
private:
	GLFWwindow * window = nullptr;
	Camera & camera;

public:
	GlBasicRenderer( int width, int height, const std::string & title, Camera & camera )
			: camera( camera )
	{
		if ( !glfwInit() )
			throw std::runtime_error( "Failed to init GLFW" );

		window = glfwCreateWindow( width, height, title.c_str(), nullptr, nullptr );
		if ( !window )
		{
			glfwTerminate();
			throw std::runtime_error( "Failed to create window" );
		}
		glfwMakeContextCurrent( window );

		if ( glewInit() != GLEW_OK )
			throw std::runtime_error( "Failed to init GLEW" );
	}

	// renders given scene configurations (basically draws pixels with glDrawPixels)
	void renderScene( const PaintScene & scene ) override
	{
		processInput();

		// render hear
		glClear( GL_COLOR_BUFFER_BIT );

		// draw triangle
		glDrawPixels( scene.width, scene.height, GL_RGB, GL_FLOAT, scene.pixels );

		// swap front and back buffers
		glfwSwapBuffers( window );

		// poll for and process events
		glfwPollEvents();
	}

	bool isAlive() override
	{
		return !glfwWindowShouldClose( window );
	}

	~GlBasicRenderer()
	{
		glfwDestroyWindow( window );
		glfwTerminate();
	}

private:
	void processInput()
	{
		if ( glfwGetKey( window, GLFW_KEY_W ) == GLFW_PRESS )
			camera.rotate( Direction::UP );
		if ( glfwGetKey( window, GLFW_KEY_S ) == GLFW_PRESS )
			camera.rotate( Direction::DOWN );
		if ( glfwGetKey( window, GLFW_KEY_A ) == GLFW_PRESS )
			camera.rotate( Direction::LEFT );
		if ( glfwGetKey( window, GLFW_KEY_D ) == GLFW_PRESS )
			camera.rotate( Direction::RIGHT );
	}
};


#endif //TRIANGLE_RAYCASTING_GLBASICRENDERER_H
