//
// Created by bartosz on 1/1/21.
//

#ifndef TRIANGLE_RAYCASTING_MATRIX3F_H
#define TRIANGLE_RAYCASTING_MATRIX3F_H

#include "Vector3f.h"


class Matrix3f
{
	static const int rowCount = 3, columnCount = 3;
	float array[rowCount * columnCount];

public:
	__host__ __device__
	Matrix3f()
	{
		for ( int i = 0; i < rowCount * columnCount; i++ )
		{
			array[i] = 0.0f;
		}
	}

	__host__ __device__
	Matrix3f( float array[] )
	{
		for ( int i = 0; i < rowCount * columnCount; i++ )
		{
			this->array[i] = array[i];
		}
	}

	__host__ __device__
	float get( int row, int col ) const
	{
		return array[index( row, col )];
	}

	__host__ __device__
	void set( int row, int col, float value )
	{
		array[index( row, col )] = value;
	}

	__host__ __device__
	Vector3f operator*( const Vector3f & v )
	{
		float product[rowCount];
		for ( int i = 0; i < rowCount; i++ )
		{
			product[i] = array[index( i, 0 )] * v.x + array[index( i, 1 )] * v.y + array[index( i, 2 )] * v.z;
		}
		return Vector3f( product[0], product[1], product[2] );
	}

	__host__ __device__
	Matrix3f operator*( const Matrix3f & other )
	{
		float productArray[rowCount * columnCount];
		for ( int i = 0; i < rowCount; i++ )
		{
			for ( int j = 0; j < columnCount; j++ )
			{
				productArray[index( i, j )] = 0.0f;
				for ( int k = 0; k < columnCount; k++ )
				{
					productArray[index( i, j )] += this->get( i, k ) * other.get( k, j );
				}
			}
		}

		return Matrix3f( productArray );
	}

private:
	__host__ __device__
	int index( int row, int col ) const
	{
		return row * rowCount + col;
	}
};


#endif //TRIANGLE_RAYCASTING_MATRIX3F_H
