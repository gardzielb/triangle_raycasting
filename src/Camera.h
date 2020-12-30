//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_CAMERA_H
#define TRIANGLE_RAYCASTING_CAMERA_H


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
	Vector3f position, target;

public:
	Camera( Vector3f target, float rotationRadius, float zoomSpeed, float rotationAngle = M_PI / 12 )
			: target( std::move( target ) ), rotationRadius( rotationRadius ), zoomSpeed( zoomSpeed ),
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

	const Vector3f & getPosition() const
	{
		return position;
	}

private:
	void updatePosition()
	{
		position = target + Vector3f(
				rotationRadius * sinf( currentAngle ), 0.0f, rotationRadius * cosf( currentAngle )
		);
	}
};


#endif //TRIANGLE_RAYCASTING_CAMERA_H
