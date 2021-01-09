//
// Created by bartosz on 12/28/20.
//

#ifndef RAYCASTING_RAYCASTER_H
#define RAYCASTING_RAYCASTER_H

#include "ScopedPtr.cuh"
#include "TriangleMesh.h"
#include "Camera.cuh"


class RayCaster
{
public:
	virtual void paintTriangleMesh( const ScopedPtr<TriangleMesh> & meshPtr, const ScopedPtr<LightSourceSet> & lightPtr,
									PaintScene & scene, const Camera & camera ) = 0;

	virtual ~RayCaster() = default;
};


#endif //RAYCASTING_RAYCASTER_H
