//
// Created by bartosz on 12/28/20.
//

#ifndef RAYCASTING_VECTOR3F_CUH
#define RAYCASTING_VECTOR3F_CUH


class Vector3f
{
public:
	float x, y, z;

	Vector3f( float x, float y, float z )
			: x( x ), y( y ), z( z )
	{}

	Vector3f() : Vector3f( 0.0f, 0.0f, 0.0f )
	{}

	Vector3f operator+( const Vector3f & other ) const
	{
		return Vector3f( x + other.x, y + other.y, z + other.z );
	}

	void operator+=( const Vector3f & other )
	{
		x += other.x;
		y += other.y;
		z += other.z;
	}

	Vector3f operator-( const Vector3f & other ) const
	{
		return Vector3f( x - other.x, y - other.y, z - other.z );
	}

	void operator-=( const Vector3f & other )
	{
		x -= other.x;
		y -= other.y;
		z -= other.z;
	}

	Vector3f cross( const Vector3f & other ) const
	{
		float xCross = y * other.z - z * other.y;
		float yCross = z * other.x - x * other.z;
		float zCross = x * other.y - y * other.x;
		return Vector3f( xCross, yCross, zCross );
	}

	float dot( const Vector3f & other ) const
	{
		return x * other.x + y * other.y + z * other.z;
	}

	float length() const
	{
		return sqrt( x * x + y * y + z * z );
	}

	Vector3f normalized() const
	{
		Vector3f v = *this;
		v.normalize();
		return v;
	}

	float distance( const Vector3f & other ) const
	{
		Vector3f v = other - *this;
		return v.length();
	}

	void normalize()
	{
		float len = length();
		if ( len < 0.00001 ) return;
		x /= len;
		y /= len;
		z /= len;
	}
};

inline Vector3f operator*( float a, const Vector3f & v )
{
	return Vector3f( a * v.x, a * v.y, a * v.z );
}


#endif //RAYCASTING_VECTOR3F_CUH
