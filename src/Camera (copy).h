////
//// Created by bartosz on 12/30/20.
////
//
//#ifndef TRIANGLE_RAYCASTING_CAMERA_H
//#define TRIANGLE_RAYCASTING_CAMERA_H
//
//
//enum class Direction
//{
//	UP, DOWN, RIGHT, LEFT, NONE
//};
//
//
//class Camera
//{
//private:
//	Vector3f position, target, direction, up, right;
//	const float speed;
//
//public:
//	Camera( float speed, Vector3f position = Vector3f( 0.0f, 0.0f, 1.0f ), Vector3f target = Vector3f() )
//			: position( std::move( position ) ), target( std::move( target ) ), speed( speed )
//	{
//		direction = (position - target).normalized();
//
//		Vector3f up1 = Vector3f( 0.0f, 1.0f, 0.0f );
//		right = up1.cross( direction ).normalized();
//
//		up = direction.cross( right ).normalized();
//	}
//
//	void rotate( Direction dir )
//	{
//		std::cout << (int) dir << "\n";
//
//		switch ( dir )
//		{
//			case Direction::UP:
//				position += speed * direction;
//				break;
//			case Direction::DOWN:
//				position -= speed * direction;
//				break;
//			case Direction::LEFT:
//				position -= speed * direction.cross( up ).normalized();
//				break;
//			case Direction::RIGHT:
//				position += speed * direction.cross( up ).normalized();
//				break;
//			default:
//				break;
//		}
//	}
//
//	const Vector3f & getPosition() const
//	{
//		return position;
//	}
//};
//
//
//#endif //TRIANGLE_RAYCASTING_CAMERA_H
