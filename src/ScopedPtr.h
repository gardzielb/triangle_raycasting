//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_SCOPEDPTR_H
#define TRIANGLE_RAYCASTING_SCOPEDPTR_H

#include "CleanupCommand.h"
#include "dependencies/helper_cuda.h"
#include "TriangleMesh.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <list>
#include <algorithm>
#include <numeric>


template<typename T>
class ScopedPtr
{
protected:
	T * object = nullptr;
	CleanupCommand * cleanupCommand = nullptr;

public:
	ScopedPtr( T * object, CleanupCommand * cleanupCommand )
	{
		this->object = object;
		this->cleanupCommand = cleanupCommand;
	}

	T * operator->()
	{
		return object;
	}

	T * getObject() const
	{
		return object;
	}

	~ScopedPtr()
	{
		cleanupCommand->execute();
		delete cleanupCommand;
	}
};


Color computeAmbientLight( const std::list<LightSource> & lightSourceList )
{
	Color specularSum, diffuseSum;
	for ( const LightSource & ls : lightSourceList )
	{
		specularSum += ls.specularLight;
		diffuseSum += ls.diffuseLight;
	}
	return (specularSum + diffuseSum) / (2 * lightSourceList.size());
}


ScopedPtr<LightSourceSet> makeCpuLightScopedPtr( const std::list<LightSource> & lightSourceList )
{
	Color ambientLight = computeAmbientLight( lightSourceList );
	LightSourceSet * lightSourceSet = new LightSourceSet( lightSourceList.size(), ambientLight );
	lightSourceSet->sources = new LightSource[lightSourceSet->count];
	std::copy( lightSourceList.begin(), lightSourceList.end(), lightSourceSet->sources );

	CleanupCommand * cleanupCommand = new CompositeCleanupCommand(
			{
					new HostCleanupCommand<LightSource>( lightSourceSet->sources, true ),
					new HostCleanupCommand<LightSourceSet>( lightSourceSet, false )
			}
	);
	return ScopedPtr<LightSourceSet>( lightSourceSet, cleanupCommand );
}


ScopedPtr<LightSourceSet> makeGpuLightScopedPtr( const std::list<LightSource> & lightSourceList )
{
	Color ambientLight = computeAmbientLight( lightSourceList );
	LightSourceSet tmpLightSourceSet( lightSourceList.size(), ambientLight );

	checkCudaErrors(
			cudaMalloc( (void **) &tmpLightSourceSet.sources, tmpLightSourceSet.count * sizeof( LightSource ) )
	);
	thrust::device_vector<LightSource> dv( lightSourceList.begin(), lightSourceList.end() );
	thrust::copy( dv.begin(), dv.end(), tmpLightSourceSet.sources );

	LightSourceSet * lightSourceSet = nullptr;
	checkCudaErrors( cudaMalloc( (void **) &lightSourceSet, sizeof( LightSourceSet ) ) );
	checkCudaErrors(
			cudaMemcpy( lightSourceSet, &tmpLightSourceSet, sizeof( LightSourceSet ), cudaMemcpyHostToDevice )
	);

	CleanupCommand * cleanupCommand = new CompositeCleanupCommand(
			{
					new GpuCleanupCommand<LightSource>( tmpLightSourceSet.sources ),
					new GpuCleanupCommand<LightSourceSet>( lightSourceSet )
			}
	);
	return ScopedPtr<LightSourceSet>( lightSourceSet, cleanupCommand );
}


ScopedPtr<TriangleMesh> makeCpuMeshScopedPtr( const std::list<Vector3f> & vertexList,
											  const std::list<Triangle> & triangleList,
											  const std::list<Color> & colorList )
{
	TriangleMesh * mesh = new TriangleMesh( triangleList.size() );
	mesh->vertices = new Vector3f[vertexList.size()];
	std::copy( vertexList.begin(), vertexList.end(), mesh->vertices );

	mesh->triangles = new Triangle[mesh->triangleCount];
	std::copy( triangleList.begin(), triangleList.end(), mesh->triangles );

	mesh->colors = new Color[mesh->triangleCount];
	std::copy( colorList.begin(), colorList.end(), mesh->colors );

	CleanupCommand * cleanupCommand = new CompositeCleanupCommand(
			{
					new HostCleanupCommand<Color>( mesh->colors, true ),
					new HostCleanupCommand<Triangle>( mesh->triangles, true ),
					new HostCleanupCommand<Vector3f>( mesh->vertices, true ),
					new HostCleanupCommand<TriangleMesh>( mesh, false )
			}
	);

	return ScopedPtr<TriangleMesh>( mesh, cleanupCommand );
}


ScopedPtr<TriangleMesh> makeGpuMeshScopedPtr( const std::list<Vector3f> & vertexList,
											  const std::list<Triangle> & triangleList,
											  const std::list<Color> & colorList )
{
	TriangleMesh tmpMesh( triangleList.size() );

	checkCudaErrors( cudaMalloc( (void **) &tmpMesh.vertices, vertexList.size() * sizeof( Vector3f ) ) );
	thrust::device_vector<Vector3f> dv( vertexList.begin(), vertexList.end() );
	thrust::copy( dv.begin(), dv.end(), tmpMesh.vertices );

	checkCudaErrors( cudaMalloc( (void **) &tmpMesh.triangles, triangleList.size() * sizeof( Triangle ) ) );
	thrust::device_vector<Triangle> tv( triangleList.begin(), triangleList.end() );
	thrust::copy( tv.begin(), tv.end(), tmpMesh.triangles );

	checkCudaErrors( cudaMalloc( (void **) &tmpMesh.colors, colorList.size() * sizeof( Color ) ) );
	thrust::device_vector<Color> cv( colorList.begin(), colorList.end() );
	thrust::copy( cv.begin(), cv.end(), tmpMesh.colors );

	TriangleMesh * mesh = nullptr;
	checkCudaErrors( cudaMalloc( (void **) &mesh, sizeof( TriangleMesh ) ) );
	checkCudaErrors( cudaMemcpy( mesh, &tmpMesh, sizeof( TriangleMesh ), cudaMemcpyHostToDevice ) );

	CleanupCommand * cleanupCommand = new CompositeCleanupCommand(
			{
					new GpuCleanupCommand<Color>( tmpMesh.colors ),
					new GpuCleanupCommand<Triangle>( tmpMesh.triangles ),
					new GpuCleanupCommand<Vector3f>( tmpMesh.vertices ),
					new GpuCleanupCommand<TriangleMesh>( mesh )
			}
	);

	return ScopedPtr<TriangleMesh>( mesh, cleanupCommand );
}


#endif //TRIANGLE_RAYCASTING_SCOPEDPTR_H
