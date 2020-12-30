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


struct TriangleMesh
{
	Vector3f * vertices = nullptr;
	Triangle * triangles = nullptr;
	int count;

	TriangleMesh( int count ) : count( count )
	{}

	TriangleMesh() : TriangleMesh( 0 )
	{}

	Vector3f & getVertex( int triangle, int vertex ) const
	{
		int index = triangles[triangle][vertex];
		return vertices[index];
	}
};


struct Color
{
	unsigned char red = 0;
	unsigned char green = 0;
	unsigned char blue = 0;
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
};


#endif //RAYCASTING_TRIANGLEMESH_H
