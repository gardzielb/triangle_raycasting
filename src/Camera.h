//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_CAMERA_H
#define TRIANGLE_RAYCASTING_CAMERA_H

#include <iostream>
#include "Matrix3f.h"

enum class Direction
{
	UP, DOWN, RIGHT, LEFT
};


// a class representing camera from which perspective rays are cast towards the screen
class Camera
{
private:
	const float rotationAngle, zoomSpeed;
	float rotationRadius;
	float currentAngle = 0.0f;
	Vector3f position, center;

public:
	Camera( Vector3f center, float rotationRadius, float zoomSpeed, float rotationAngle = M_PI / 36 )
			: center( std::move( center ) ), rotationRadius( rotationRadius ), zoomSpeed( zoomSpeed ),
			  rotationAngle( rotationAngle )
	{
		updatePosition();
	}

	// camera can be rotated around the [0,1,0] axis
	void rotate( Direction dir )
	{
		switch ( dir )
		{
			case Direction::UP:
				rotationRadius -= zoomSpeed;
				break;
			case Direction::DOWN:
				rotationRadius += zoomSpeed;
				break;
			case Direction::LEFT:
				currentAngle -= rotationAngle;
				break;
			case Direction::RIGHT:
				currentAngle += rotationAngle;
				break;
		}

		updatePosition();
	}

	__host__ __device__
	const Vector3f & getPosition() const
	{
		return position;
	}

	// returns a direction of a ray pointing from the camera towards the given point on a screen;
	// the screen is placed opposite to the camera, so that a ray passing through (0,0,0) is perpendicular to the screen
	// and points at (0,0) point on the screen
	__host__ __device__
	Vector3f emitRay( float x, float y ) const
	{
		Matrix3f rotationMatrix;
		rotationMatrix.set( 0, 0, cosf( currentAngle ) );
		rotationMatrix.set( 0, 2, sinf( currentAngle ) );
		rotationMatrix.set( 1, 1, 1.0f );
		rotationMatrix.set( 2, 0, -sinf( currentAngle ) );
		rotationMatrix.set( 2, 2, cosf( currentAngle ) );
		return rotationMatrix * Vector3f( x, y, -1.0f );
	}

private:
	void updatePosition()
	{
		position = center + Vector3f(
				rotationRadius * sinf( currentAngle ), 0.0f, rotationRadius * cosf( currentAngle )
		);
	}
};


#endif //TRIANGLE_RAYCASTING_CAMERA_H
