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

// MeshLoader implementation using Assimp library to load models from .obj files
class ObjMeshLoader : public MeshLoader
{
private:
	std::string path;
	Color color;
	float shininess;

public:
	ObjMeshLoader( const std::string & path, const Color & color, float shininess )
			: path( path ), color( color ), shininess( shininess )
	{}

	ScopedPtr<TriangleMesh> loadMesh( DestMemoryKind kind ) override
	{
		std::list<Vector3f> vertexList;
		std::list<Triangle> triangleList;

		// read file via ASSIMP
		Assimp::Importer importer;
		const aiScene * scene = importer.ReadFile( path, aiProcess_Triangulate );
		// check for errors
		if ( !scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode ) // if is Not Zero
		{
			std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << "\n";
			return createMesh( kind, vertexList, triangleList );
		}

		for ( int i = 0; i < scene->mNumMeshes; i++ )
		{
			processMesh( scene->mMeshes[i], vertexList, triangleList );
		}

		std::cout << "TRIANGLES: " << triangleList.size() << ", VERTICES: " << vertexList.size() << "\n";
		return createMesh( kind, vertexList, triangleList );
	}

private:
	ScopedPtr<TriangleMesh> createMesh( DestMemoryKind kind, const std::list<Vector3f> & vertexList,
										const std::list<Triangle> & triangleList )
	{
		if ( kind == DestMemoryKind::CPU )
			return makeCpuMeshScopedPtr( vertexList, triangleList, color, shininess );
		return makeGpuMeshScopedPtr( vertexList, triangleList, color, shininess );
	}

	void processMesh( const aiMesh * mesh, std::list<Vector3f> & vertices, std::list<Triangle> & triangles )
	{
		for ( int i = 0; i < mesh->mNumFaces; i++ )
		{
			aiFace & face = mesh->mFaces[i];
			if ( face.mNumIndices > 3 )
				std::cout << "XD\n";

			triangles.push_back(
					{
							(int) face.mIndices[0], (int) face.mIndices[1], (int) face.mIndices[2]
					}
			);
		}

		for ( int i = 0; i < mesh->mNumVertices; i++ )
		{
			vertices.push_back( { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z } );
		}
	}
};



