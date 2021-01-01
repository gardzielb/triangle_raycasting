//
// Created by bartosz on 12/28/20.
//

#include "raycasting.cuh"


__host__ __device__
bool rayIntersectsTriangle( const Vector3f & rayOrigin, const Vector3f & rayVector,
							const Vector3f & v0, const Vector3f & v1, const Vector3f & v2,
							Vector3f * outIntersectionPoint )
{
	const float EPSILON = 0.0000001;
	Vector3f edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = v1 - v0;
	edge2 = v2 - v0;
	h = rayVector.cross( edge2 );
	a = edge1.dot( h );
	if ( a > -EPSILON && a < EPSILON )
		return false;    // This ray is parallel to this triangle.
	f = 1.0 / a;
	s = rayOrigin - v0;
	u = f * s.dot( h );
	if ( u < 0.0 || u > 1.0 )
		return false;
	q = s.cross( edge1 );
	v = f * rayVector.dot( q );
	if ( v < 0.0 || u + v > 1.0 )
		return false;
	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * edge2.dot( q );
	if ( t > EPSILON ) // ray intersection
	{
		if ( outIntersectionPoint )
			*outIntersectionPoint = rayOrigin + t * rayVector;
		return true;
	} else // This means that there is a line intersection but not a ray intersection.
		return false;
}


__host__ __device__
void doRayCasting( int x, int y, const TriangleMesh * mesh, PaintScene * scene, const Camera & camera )
{
	scene->setPixel( x, y, { 0, 0, 0 } );

	float targetX = (2 * (float) x) / scene->width - 1;
	float targetY = (2 * (float) y) / scene->height - 1;
	Vector3f rayVector = camera.emitRay( targetX, targetY );
	float minDist = MAXFLOAT;

	for ( int i = 0; i < mesh->triangleCount; i++ )
	{
		Vector3f intersection;
		bool isHit = rayIntersectsTriangle(
				camera.getPosition(), rayVector, mesh->getVertex( i, 0 ), mesh->getVertex( i, 1 ),
				mesh->getVertex( i, 2 ), &intersection
		);
		if ( !isHit ) continue;

		scene->setPixel( x, y, { 255, 0, 0 } );

		float dist = intersection.distance( camera.getPosition() );
		if ( dist < minDist )
		{
			minDist = dist;
			scene->setPixel( x, y, mesh->colors[i] );
		}
	}
}