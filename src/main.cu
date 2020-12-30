#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "CpuRayCaster.h"
#include "IndexMeshLoader.h"


int main()
{
	if ( !glfwInit() )
		throw std::runtime_error( "Failed to init GLFW" );

	int width = 800, height = 800;
	GLFWwindow * window = glfwCreateWindow( width, height, "Raycasting", nullptr, nullptr );
	if ( !window )
	{
		glfwTerminate();
		throw std::runtime_error( "Failed to create window" );
	}
	glfwMakeContextCurrent( window );

	if ( glewInit() != GLEW_OK )
		throw std::runtime_error( "Failed to init GLEW" );

	IndexMeshLoader meshLoader( "../vertices.txt", "../triangles.txt" );
	TriangleMesh mesh = meshLoader.loadMesh();

	PaintScene scene;
	scene.height = height;
	scene.width = width;
	scene.pixels = new Color[scene.width * scene.height];

	CpuRayCaster rayCaster;
	rayCaster.paintTriangleMesh( mesh, scene, Vector3f( scene.width / 2, scene.height / 2, -400.0f ) );

	while ( !glfwWindowShouldClose( window ) )
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

	delete[] scene.pixels;

	glDisableClientState( GL_COLOR_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
	glfwDestroyWindow( window );
	glfwTerminate();

	return 0;
}
