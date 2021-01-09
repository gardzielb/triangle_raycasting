//
// Created by bartosz on 12/30/20.
//

#ifndef RAYCASTING_INDEXMESHLOADER_H
#define RAYCASTING_INDEXMESHLOADER_H

#include "MeshLoader.h"
#include "readFileUtils.h"


class IndexMeshLoader : public MeshLoader
{
private:
	std::string vertexFile, triangleFile, colorFile;

public:
	IndexMeshLoader( const std::string & vertexFile, const std::string & triangleFile, const std::string & colorFile )
			: vertexFile( vertexFile ), triangleFile( triangleFile ), colorFile( colorFile )
	{}

	ScopedPtr<TriangleMesh> loadMesh( DestMemoryKind kind ) override
	{
		std::list<Vector3f> vertexList;
		std::list<Triangle> triangleList;
		std::list<Color> colorList;

		std::ifstream vertexStream( vertexFile );
		std::string line;
		while ( getline( vertexStream, line ) )
		{
			vertexList.push_back( readVertex( line ) );
		}
		vertexStream.close();

		std::ifstream triangleStream( triangleFile );
		while ( getline( triangleStream, line ) )
		{
			triangleList.push_back( readTriangle( line ) );
		}
		triangleStream.close();

		std::ifstream colorStream( colorFile );
		while ( getline( colorStream, line ) )
		{
			colorList.push_back( readColor( line ) );
		}
		colorStream.close();

		if ( kind == DestMemoryKind::CPU )
			return makeCpuMeshScopedPtr( vertexList, triangleList, colorList );
		return makeGpuMeshScopedPtr( vertexList, triangleList, colorList );
	}
};


#endif //RAYCASTING_INDEXMESHLOADER_H
