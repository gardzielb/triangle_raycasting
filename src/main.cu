#include <iostream>
#include "CpuRayCaster.h"
#include "IndexMeshLoader.h"
#include "GlBasicRenderer.h"
#include "Camera.h"


int main()
{
	int width = 800, height = 800;

	IndexMeshLoader meshLoader( "../vertices.txt", "../triangles.txt", "../colors.txt" );
	TriangleMesh mesh = meshLoader.loadMesh();

	PaintScene scene;
	scene.height = height;
	scene.width = width;
	scene.pixels = new Color[scene.width * scene.height];

	CpuRayCaster rayCaster;
	Camera camera( Vector3f( width / 2, height / 2, 0.0f ), 1000.0f, 150.0f );
	GlBasicRenderer renderer( width, height, "Raycasting", camera );

	while ( renderer.isAlive() )
	{
		scene.clear();
		rayCaster.paintTriangleMesh( mesh, scene, camera.getPosition() );
		renderer.renderScene( scene );
	}

	return 0;
}
