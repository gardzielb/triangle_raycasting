//
// Created by bartosz on 12/29/20.
//

#ifndef RAYCASTING_CPURAYCASTER_H
#define RAYCASTING_CPURAYCASTER_H

#include "RayCaster.h"
#include "utils.cuh"


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

				for ( int i = 0; i < mesh.triangleCount; i++ )
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
						scene[y][x] = mesh.colors[i];
					}
				}
			}
		}
	}
};


#endif //RAYCASTING_CPURAYCASTER_H
