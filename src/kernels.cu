//
// Created by bartosz on 12/30/20.
//

#include "kernels.cuh"
#include "raycasting.cuh"


__global__
void rayCastingKernel( const TriangleMesh * mesh, PaintScene * scene, const Camera * camera,
					   const LightSourceSet * lightSources )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if ( x >= scene->width ) return;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( y >= scene->height ) return;

	doRayCasting( x, y, mesh, scene, *camera, *lightSources );
}