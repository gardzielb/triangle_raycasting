//
// Created by bartosz on 12/29/20.
//
#ifndef RAYCASTING_TRIANGLEMESH_H
#define RAYCASTING_TRIANGLEMESH_H

#include "Vector3f.cuh"


struct Triangle
{
	int vertices[3];

	int operator[]( int i ) const
	{
		return vertices[i];
	}
};


struct Color
{
	unsigned char red = 0;
	unsigned char green = 0;
	unsigned char blue = 0;
};


struct TriangleMesh
{
	Vector3f * vertices = nullptr;
	Triangle * triangles = nullptr;
	Color * colors = nullptr; // one color for each triangle
	int triangleCount;

	TriangleMesh( int triangleCount ) : triangleCount( triangleCount )
	{}

	TriangleMesh() : TriangleMesh( 0 )
	{}

	Vector3f & getVertex( int triangle, int vertex ) const
	{
		int index = triangles[triangle][vertex];
		return vertices[index];
	}

	~TriangleMesh()
	{
		if ( vertices )
			delete[] vertices;
		if ( triangles )
			delete[] triangles;
		if ( colors )
			delete[] colors;
	}
};


struct PaintScene
{
	Color * pixels;
	int width;
	int height;

	Color * operator[]( int y )
	{
		return pixels + y * width;
	}

	~PaintScene()
	{
		if ( pixels )
			delete[] pixels;
	}
};


#endif //RAYCASTING_TRIANGLEMESH_H
