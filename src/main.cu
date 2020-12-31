#include <iostream>
#include "CpuRayCaster.h"
#include "IndexMeshLoader.h"
#include "GlBasicRenderer.h"
#include "Camera.h"
#include "GpuRayCaster.cuh"


static DestMemoryKind enumValueOf( const std::string & str )
{
	return str == "gpu" ? DestMemoryKind::GPU : DestMemoryKind::CPU;
}


static RayCaster * createRayCaster( DestMemoryKind kind, int width, int height )
{
	if ( kind == DestMemoryKind::GPU )
		return new GpuRayCaster( width, height );
	return new CpuRayCaster;
}


int main( int argc, char ** argv )
{
	int width = 800, height = 800;
	DestMemoryKind kind = enumValueOf( argv[1] );

	IndexMeshLoader meshLoader( "../vertices.txt", "../triangles.txt", "../colors.txt" );
	auto meshPtr = meshLoader.loadMesh( kind );

	PaintScene scene;
	scene.height = height;
	scene.width = width;
	scene.pixels = new Color[scene.width * scene.height];

	RayCaster * rayCaster = createRayCaster( kind, width, height );

	Camera camera( Vector3f( width / 2, height / 2, 0.0f ), 1000.0f, 150.0f );
	GlBasicRenderer renderer( width, height, "Raycasting", camera );

	while ( renderer.isAlive() )
	{
		rayCaster->paintTriangleMesh( meshPtr, scene, camera.getPosition() );
		renderer.renderScene( scene );
	}

	delete rayCaster;
	delete[] scene.pixels;
	return 0;
}
