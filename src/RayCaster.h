//
// Created by bartosz on 12/28/20.
//

#ifndef RAYCASTING_RAYCASTER_H
#define RAYCASTING_RAYCASTER_H

#include "TriangleMeshScopedPtr.cuh"
#include "TriangleMesh.h"


class RayCaster
{
public:
	virtual void paintTriangleMesh( const TriangleMeshScopedPtr & meshPtr, PaintScene & scene,
									const Vector3f & cameraPos ) = 0;

	virtual ~RayCaster() = default;
};


#endif //RAYCASTING_RAYCASTER_H
