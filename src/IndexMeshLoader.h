//
// Created by bartosz on 12/30/20.
//

#ifndef RAYCASTING_INDEXMESHLOADER_H
#define RAYCASTING_INDEXMESHLOADER_H

#include "MeshLoader.h"


class IndexMeshLoader : public MeshLoader
{
private:
	std::string vertexFile, triangleFile;

public:
	IndexMeshLoader( const std::string & vertexFile, const std::string & triangleFile )
			: vertexFile( vertexFile ), triangleFile( triangleFile )
	{}

	TriangleMesh loadMesh() override
	{
		std::list<Vector3f> vertexList;
		std::list<Triangle> triangleList;

		std::ifstream vertexStream( vertexFile );
		std::string line;
		while ( getline( vertexStream, line ) )
		{
			vertexList.push_back( readVertex( line ) );
		}

		std::ifstream triangleStream( triangleFile );
		while ( getline( triangleStream, line ) )
		{
			triangleList.push_back( readTriangle( line ) );
		}

		return createMesh( vertexList, triangleList );
	}

private:
	TriangleMesh createMesh( const std::list<Vector3f> & vertexList, const std::list<Triangle> & triangleList )
	{
		TriangleMesh mesh( triangleList.size() );
		mesh.vertices = new Vector3f[vertexList.size()];
		std::copy( vertexList.begin(), vertexList.end(), mesh.vertices );

		mesh.triangles = new Triangle[mesh.count];
		std::copy( triangleList.begin(), triangleList.end(), mesh.triangles );

		return mesh;
	}

	Triangle readTriangle( const std::string & line )
	{
		int space1 = line.find( ' ' );
		if ( space1 == std::string::npos )
			throw std::runtime_error( "Invalid line" );
		std::string aStr = line.substr( 0, space1 );
		int a = std::stoi( aStr );

		int space2 = line.find( ' ', space1 + 1 );
		if ( space2 == std::string::npos )
			throw std::runtime_error( "Invalid line" );
		std::string bStr = line.substr( space1 + 1, space2 );
		int b = std::stoi( bStr );

		std::string cStr = line.substr( space2 + 1 );
		int c = std::stoi( cStr );

		return { a, b, c };
	}

	Vector3f readVertex( const std::string & line )
	{
		int space1 = line.find( ' ' );
		if ( space1 == std::string::npos )
			throw std::runtime_error( "Invalid line" );
		std::string xStr = line.substr( 0, space1 );
		float x = std::stof( xStr );

		int space2 = line.find( ' ', space1 + 1 );
		if ( space2 == std::string::npos )
			throw std::runtime_error( "Invalid line" );
		std::string yStr = line.substr( space1 + 1, space2 );
		float y = std::stof( yStr );

		std::string zStr = line.substr( space2 + 1 );
		float z = std::stof( zStr );

		return Vector3f( x, y, z );
	}
};


#endif //RAYCASTING_INDEXMESHLOADER_H
