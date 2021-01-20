#include <iostream>
#include "CpuRayCaster.h"
#include "IndexMeshLoader.h"
#include "GlBasicRenderer.h"
#include "Camera.h"
#include "GpuRayCaster.h"
#include "SimpleLightsLoader.h"
#include "ObjMeshLoader.h"


static DestMemoryKind enumValueOf( const std::string & str )
{
	return str == "gpu" ? DestMemoryKind::GPU : DestMemoryKind::CPU;
}


static RayCaster * createRayCaster( DestMemoryKind kind, int width, int height )
{
	if ( kind == DestMemoryKind::GPU )
		return new GpuRayCaster( width, height );
	return new CpuRayCaster();
}


int main( int argc, char ** argv )
{
	int width = 800, height = 800;
	DestMemoryKind kind = enumValueOf( argv[1] );
	std::string model = argv[2];

//	IndexMeshLoader meshLoader(
//			"../models/" + model + "/vertices.txt", "../models/" + model + "/triangles.txt",
//			"../models/" + model + "/colors.txt"
//	);
	ObjMeshLoader meshLoader( "../models/" + model + ".obj" );
	auto meshPtr = meshLoader.loadMesh( kind );

	PaintScene scene;
	scene.height = height;
	scene.width = width;
	scene.pixels = new Color[scene.width * scene.height];

	SimpleLightsLoader lightsLoader( "../lights.txt" );
	auto lightsPtr = lightsLoader.loadLights( kind );

	RayCaster * rayCaster = createRayCaster( kind, width, height );

	Camera camera( Vector3f( 0.0f, 0.0f, 0.0f ), 3.0f, 0.1f );
	GlBasicRenderer renderer( width, height, "Raycasting", camera );

	while ( renderer.isAlive() )
	{
		rayCaster->paintTriangleMesh( meshPtr, lightsPtr, scene, camera );
		renderer.renderScene( scene );
	}

	delete rayCaster;
	delete[] scene.pixels;
	return 0;
}
