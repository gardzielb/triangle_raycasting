//
// Created by bartosz on 12/29/20.
//

#ifndef RAYCASTING_CPURAYCASTER_H
#define RAYCASTING_CPURAYCASTER_H

#include "RayCaster.h"
#include "utils.cuh"


Color colors[] = {
		{ 255, 0,   0 },
		{ 0,   255, 0 },
		{ 0,   0,   255 }
};


class CpuRayCaster : public RayCaster
{
public:
	void paintTriangleMesh( const TriangleMesh & mesh, PaintScene & scene, const Vector3f & cameraPos ) override
	{
		for ( int x = 0; x < scene.width; x++ )
		{
			for ( int y = 0; y < scene.height; y++ )
			{
				Vector3f rayVector = Vector3f( x, y, 1.0f ) - cameraPos;
				float minDist = MAXFLOAT;

				for ( int i = 0; i < mesh.count; i++ )
				{
					Vector3f intersection;
					bool isHit = rayIntersectsTriangle(
							cameraPos, rayVector, mesh.getVertex( i, 0 ), mesh.getVertex( i, 1 ),
							mesh.getVertex( i, 2 ), &intersection
					);
					if ( !isHit ) continue;

					float dist = intersection.distance( cameraPos );
					if ( dist < minDist )
					{
						minDist = dist;
						scene[y][x] = colors[i % 3];
					}
				}
			}
		}
	}
};


#endif //RAYCASTING_CPURAYCASTER_H
