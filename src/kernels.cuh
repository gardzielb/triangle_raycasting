//
// Created by bartosz on 12/30/20.
//

#include "Vector3f.cuh"
#include "TriangleMesh.h"
#include "Camera.cuh"

#ifndef TRIANGLE_RAYCASTING_KERNELS_CUH
#define TRIANGLE_RAYCASTING_KERNELS_CUH

__global__
void rayCastingKernel( const TriangleMesh * mesh, PaintScene * scene, const Camera * camera,
					   const LightSourceSet * lightSources );

#endif //TRIANGLE_RAYCASTING_KERNELS_CUH