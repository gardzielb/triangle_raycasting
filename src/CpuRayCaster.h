//
// Created by bartosz on 12/29/20.
//

#ifndef RAYCASTING_CPURAYCASTER_H
#define RAYCASTING_CPURAYCASTER_H

#include "RayCaster.h"
#include "raycasting.cuh"


// simple CPU impementation of a RayCaster
class CpuRayCaster : public RayCaster
{
public:
	void paintTriangleMesh( const ScopedPtr<TriangleMesh> & meshPtr, const ScopedPtr<LightSourceSet> & lightPtr,
							PaintScene & scene, const Camera & camera ) override
	{
		for ( int x = 0; x < scene.width; x++ )
		{
			for ( int y = 0; y < scene.height; y++ )
			{
				doRayCasting( x, y, meshPtr.getObject(), &scene, camera, *lightPtr.getObject() );
			}
		}
	}
};


#endif //RAYCASTING_CPURAYCASTER_H
