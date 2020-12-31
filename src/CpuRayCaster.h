//
// Created by bartosz on 12/29/20.
//

#ifndef RAYCASTING_CPURAYCASTER_H
#define RAYCASTING_CPURAYCASTER_H

#include "RayCaster.h"
#include "raycasting.cuh"


class CpuRayCaster : public RayCaster
{
public:
	void paintTriangleMesh( const TriangleMeshScopedPtr & meshPtr, PaintScene & scene,
							const Vector3f & cameraPos ) override
	{
		for ( int x = 0; x < scene.width; x++ )
		{
			for ( int y = 0; y < scene.height; y++ )
			{
				doRayCasting( x, y, meshPtr.getMesh(), &scene, cameraPos );
			}
		}
	}
};


#endif //RAYCASTING_CPURAYCASTER_H
