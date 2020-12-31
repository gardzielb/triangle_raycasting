//
// Created by bartosz on 12/29/20.
//
#ifndef RAYCASTING_TRIANGLEMESH_H
#define RAYCASTING_TRIANGLEMESH_H

#include "Vector3f.cuh"


struct Triangle
{
	int vertices[3];

	__host__ __device__
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

	__host__ __device__
	Vector3f & getVertex( int triangle, int vertex ) const
	{
		int index = triangles[triangle][vertex];
		return vertices[index];
	}
};


struct PaintScene
{
	Color * pixels;
	int width;
	int height;

	__host__ __device__
	void setPixel( int x, int y, const Color & color )
	{
		pixels[y * width + x] = color;
	}
};


#endif //RAYCASTING_TRIANGLEMESH_H
