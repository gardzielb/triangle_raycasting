#include <iostream>
#include "CpuRayCaster.h"
#include "IndexMeshLoader.h"
#include "GlBasicRenderer.h"
#include "Camera.cuh"
#include "GpuRayCaster.cuh"


static DestMemoryKind enumValueOf( const std::string & str )
{
	return str == "gpu" ? DestMemoryKind::GPU : DestMemoryKind::CPU;
}


static RayCaster * createRayCaster( DestMemoryKind kind, int width, int height, LightSourceSet & lightSourceSet )
{
	if ( kind == DestMemoryKind::GPU )
		return new GpuRayCaster( width, height, lightSourceSet );
	return new CpuRayCaster( lightSourceSet );
}


int main( int argc, char ** argv )
{
	int width = 800, height = 800;
	DestMemoryKind kind = enumValueOf( argv[1] );
	std::string model = argv[2];

	IndexMeshLoader meshLoader(
			"../models/" + model + "/vertices.txt", "../models/" + model + "/triangles.txt",
			"../models/" + model + "/colors.txt"
	);
	auto meshPtr = meshLoader.loadMesh( kind );

	PaintScene scene;
	scene.height = height;
	scene.width = width;
	scene.pixels = new Color[scene.width * scene.height];

	LightSourceSet lightSources( 1, Color( 255, 255, 255 ) );
	lightSources.sources = new LightSource[lightSources.count];
	lightSources[0] = LightSource(
			Vector3f( 0.0f, 0.5f, 1.0f ), Color( 255, 255, 255 ), Color( 255, 255, 255 )
	);

	RayCaster * rayCaster = createRayCaster( kind, width, height, lightSources );

	Camera camera( Vector3f( 0.0f, 0.0f, 0.0f ), 1.0f, 0.1f );
	GlBasicRenderer renderer( width, height, "Raycasting", camera );

	while ( renderer.isAlive() )
	{
		rayCaster->paintTriangleMesh( meshPtr, scene, camera );
		renderer.renderScene( scene );
	}

	delete rayCaster;
	delete[] lightSources.sources;
	delete[] scene.pixels;
	return 0;
}
