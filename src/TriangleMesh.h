//
// Created by bartosz on 12/29/20.
//
#ifndef RAYCASTING_TRIANGLEMESH_H
#define RAYCASTING_TRIANGLEMESH_H

#include "Vector3f.h"
#include "Color.h"


enum class DestMemoryKind
{
	CPU, GPU
};


// holds indices of the triangle vertices stored in one array in TriangleMesh struct
struct Triangle
{
	int vertices[3];

	__host__ __device__
	int operator[]( int i ) const
	{
		return vertices[i];
	}
};


// holds array of vertices and triangles, provides some access utilities
struct TriangleMesh
{
	Vector3f * vertices = nullptr;
	Triangle * triangles = nullptr;
	Color color; // one color for each triangle
	int triangleCount;
	float shininess;

	TriangleMesh( int triangleCount ) : triangleCount( triangleCount ), shininess( 32.0f )
	{}

	TriangleMesh( int triangleCount, const Color & color, float shininess )
			: triangleCount( triangleCount ), color( color ), shininess( 32.0f )
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


// utility wrapper for pixel array
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


// provides ambient light for the light source set and the access operator
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
