//
// Created by bartosz on 12/28/20.
//

#ifndef RAYCASTING_UTILS_CUH
#define RAYCASTING_UTILS_CUH

#include "Vector3f.cuh"

bool rayIntersectsTriangle( const Vector3f & rayOrigin, const Vector3f & rayVector,
							const Vector3f & v0, const Vector3f & v1, const Vector3f & v2,
							Vector3f * outIntersectionPoint );

#endif //RAYCASTING_UTILS_CUH
