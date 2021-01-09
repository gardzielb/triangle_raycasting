//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_CAMERA_CUH
#define TRIANGLE_RAYCASTING_CAMERA_CUH

#include <iostream>
#include "Matrix3f.cuh"

enum class Direction
{
	UP, DOWN, RIGHT, LEFT
};


class Camera
{
private:
	const float rotationAngle, zoomSpeed;
	float rotationRadius;
	float currentAngle = 0.0f;
	Vector3f position, center;

public:
	Camera( Vector3f center, float rotationRadius, float zoomSpeed, float rotationAngle = M_PI / 72 )
			: center( std::move( center ) ), rotationRadius( rotationRadius ), zoomSpeed( zoomSpeed ),
			  rotationAngle( rotationAngle )
	{
		updatePosition();
	}

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


#endif //TRIANGLE_RAYCASTING_CAMERA_CUH
