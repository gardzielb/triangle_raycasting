//
// Created by bartosz on 1/20/21.
//

#pragma once

#include "MeshLoader.h"
#include <list>
#include "Vector3f.h"
#include "Color.h"
#include "TriangleMesh.h"
#include "ScopedPtr.h"
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>


class ObjMeshLoader : public MeshLoader
{
private:
	std::string path;

public:
	ObjMeshLoader( const std::string & path )
			: path( path )
	{}

	ScopedPtr<TriangleMesh> loadMesh( DestMemoryKind kind ) override
	{
		std::list<Vector3f> vertexList;
		std::list<Triangle> triangleList;
		std::list<Color> colorList;

		// read file via ASSIMP
		Assimp::Importer importer;
		const aiScene * scene = importer.ReadFile(
				path,
				aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace
		);
		// check for errors
		if ( !scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode ) // if is Not Zero
		{
			std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << "\n";
			return createMesh( kind, vertexList, triangleList, colorList );
		}

		for ( int i = 0; i < scene->mNumMeshes; i++ )
		{
			processMesh( scene->mMeshes[i], vertexList, triangleList, colorList );
		}

		std::cout << "TRIANGLES: " << triangleList.size() << ", VERTICES: " << vertexList.size() << "\n";
		return createMesh( kind, vertexList, triangleList, colorList );
	}

private:
	ScopedPtr<TriangleMesh> createMesh( DestMemoryKind kind, const std::list<Vector3f> & vertexList,
										const std::list<Triangle> & triangleList, const std::list<Color> & colorList )
	{
		if ( kind == DestMemoryKind::CPU )
			return makeCpuMeshScopedPtr( vertexList, triangleList, colorList );
		return makeGpuMeshScopedPtr( vertexList, triangleList, colorList );
	}

	void processMesh( const aiMesh * mesh, std::list<Vector3f> & vertices, std::list<Triangle> & triangles,
					  std::list<Color> & colors )
	{
		for ( int i = 0; i < mesh->mNumFaces; i++ )
		{
			aiFace & face = mesh->mFaces[i];
			triangles.push_back(
					{
							(int) face.mIndices[0], (int) face.mIndices[1], (int) face.mIndices[2]
					}
			);
			colors.emplace_back( 255, 255, 255 );
		}

		for ( int i = 0; i < mesh->mNumVertices; i++ )
		{
			vertices.push_back( { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z } );
		}
	}
};



