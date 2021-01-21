//
// Created by bartosz on 12/28/20.
//

#include "raycasting.cuh"


// checks whether ray intersects with given triangle using Möller–Trumbore algorithm
// returns boolean value and if true, stores intersection in outIntersectionPoint
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
	}
	else // This means that there is a line intersection but not a ray intersection.
		return false;
}


// computes normal vector in given point of a given triangle
// rayOrigin is necessary to choose the normal 'closer' to a camera, not the one pointing inside the model
__host__ __device__
Vector3f normalVector( const Vector3f & rayOrigin, const Vector3f & point, int triangleIndex,
					   const TriangleMesh * mesh )
{
	Vector3f p0 = mesh->getVertex( triangleIndex, 0 );
	Vector3f u = mesh->getVertex( triangleIndex, 1 ) - p0;
	Vector3f v = mesh->getVertex( triangleIndex, 2 ) - p0;
	Vector3f normalVector = u.cross( v ).normalized();

	Vector3f rayVector = rayOrigin - point;
	if ( rayVector.normalized().dot( normalVector ) < 0 )
		normalVector = (-1) * normalVector;

	return normalVector;
}


// computes the pixel color according to Phong reflection model
__host__ __device__
Color phongReflectionColor( const Vector3f & point, const Vector3f & cameraPos, const Vector3f & normalVector,
							const LightSourceSet & lightSources, const Color & color, float shininess )
{
	Vector3f cameraVector = (cameraPos - point).normalized();
	Color outColor = lightSources.ambientLight;

	for ( int i = 0; i < lightSources.count; i++ )
	{
		Vector3f lightVector = (lightSources[i].position - point).normalized();
		float lnDot = lightVector.dot( normalVector );
		Vector3f reflectionVector = 2 * lnDot * normalVector - lightVector;
		float rvDot = reflectionVector.dot( cameraVector );

		Color d = lnDot * lightSources[i].diffuseLight;
		Color s = powf( rvDot, shininess ) * lightSources[i].specularLight;
		outColor += (d + s);
	}

	outColor.shrink();
	return color * outColor;
}


// performs raycasting using naive linear algorithm searching all the triangles
__host__ __device__
void doRayCasting( int x, int y, const TriangleMesh * mesh, PaintScene * scene, const Camera & camera,
				   const LightSourceSet & lightSources )
{
	scene->setPixel( x, y, Color() );

	float targetX = (2 * (float) x) / scene->width - 1;
	float targetY = (2 * (float) y) / scene->height - 1;
	Vector3f rayVector = camera.emitRay( targetX, targetY );

	float minDist = MAXFLOAT;
	Vector3f closestIntersection;
	int triangleIndex = -1;

	for ( int i = 0; i < mesh->triangleCount; i++ )
	{
		Vector3f intersection;
		bool isHit = rayIntersectsTriangle(
				camera.getPosition(), rayVector, mesh->getVertex( i, 0 ), mesh->getVertex( i, 1 ),
				mesh->getVertex( i, 2 ), &intersection
		);
		if ( !isHit ) continue;

		// search for the closest intersection
		float dist = intersection.distance( camera.getPosition() );
		if ( dist < minDist )
		{
			minDist = dist;
			triangleIndex = i;
			closestIntersection = intersection;
		}
	}

	// compute color only once for the closest intersection if any found
	if ( triangleIndex >= 0 )
	{
		Vector3f vNormal = normalVector( camera.getPosition(), closestIntersection, triangleIndex, mesh );
		Color color = phongReflectionColor(
				closestIntersection, camera.getPosition(), vNormal, lightSources, mesh->color, mesh->shininess
		);
		scene->setPixel( x, y, color );
	}
}