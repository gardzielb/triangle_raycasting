#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "CpuRayCaster.h"
#include "IndexMeshLoader.h"
#include "GlBasicRenderer.h"


int main()
{
	int width = 800, height = 800;

	IndexMeshLoader meshLoader( "../vertices.txt", "../triangles.txt" );
	TriangleMesh mesh = meshLoader.loadMesh();

	PaintScene scene;
	scene.height = height;
	scene.width = width;
	scene.pixels = new Color[scene.width * scene.height];

	CpuRayCaster rayCaster;
	rayCaster.paintTriangleMesh( mesh, scene, Vector3f( scene.width / 2, scene.height / 2, -400.0f ) );

	GlBasicRenderer renderer( width, height, "Raycasting" );

	while ( renderer.isAlive() )
	{
		renderer.renderScene( scene );
	}

	delete[] scene.pixels;

	return 0;
}
