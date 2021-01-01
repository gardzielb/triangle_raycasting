//
// Created by bartosz on 12/29/20.
//
#ifndef RAYCASTING_TRIANGLEMESH_H
#define RAYCASTING_TRIANGLEMESH_H

#include "Vector3f.cuh"
#include "Color.cuh"


struct Triangle
{
	int vertices[3];

	__host__ __device__
	int operator[]( int i ) const
	{
		return vertices[i];
	}
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


struct LightSource
{
	Vector3f position;
	Color specularLight, diffuseLight;

public:
	LightSource( Vector3f position, Color specularLight, Color diffuseLight )
			: position( std::move( position ) ), specularLight( std::move( specularLight ) ),
			  diffuseLight( std::move( diffuseLight ) )
	{}

	LightSource() : LightSource( Vector3f(), Color(), Color() )
	{}
};


struct LightSourceSet
{
	LightSource * sources = nullptr;
	Color ambientLight;
	int count = 0;

	LightSourceSet()
	{}

	LightSourceSet( int count, Color ambientLight )
			: count( count ), ambientLight( std::move( ambientLight ) )
	{}

	__host__ __device__
	LightSource & operator[]( int i ) const
	{
		return sources[i];
	}
};


#endif //RAYCASTING_TRIANGLEMESH_H
