//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_GLBASICRENDERER_H
#define TRIANGLE_RAYCASTING_GLBASICRENDERER_H

#include "Renderer.h"
#include "GL/glew.h"
#include <GLFW/glfw3.h>


class GlBasicRenderer : public Renderer
{
private:
	GLFWwindow * window = nullptr;

public:
	GlBasicRenderer( int width, int height, const std::string & title )
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

	void renderScene( const PaintScene & scene ) override
	{
		// render hear
		glClear( GL_COLOR_BUFFER_BIT );

		// draw triangle
		glDrawPixels( scene.width, scene.height, GL_RGB, GL_UNSIGNED_BYTE, scene.pixels );

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
};


#endif //TRIANGLE_RAYCASTING_GLBASICRENDERER_H
