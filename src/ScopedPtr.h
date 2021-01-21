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


// a custom smart pointer behaving mostly like std::unique_ptr;
// capable of freeing memory on GPU with proper CleanupCommand
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


// utility function for building LightSourceSet
Color computeAmbientLight( const std::list<LightSource> & lightSourceList, float ambientStrength )
{
	Color specularSum, diffuseSum;
	for ( const LightSource & ls : lightSourceList )
	{
		specularSum += ls.specularLight;
		diffuseSum += ls.diffuseLight;
	}
	return ambientStrength * (specularSum + diffuseSum) / (2 * lightSourceList.size());
}


ScopedPtr<LightSourceSet> makeCpuLightScopedPtr( const std::list<LightSource> & lightSourceList, float ambientStrength )
{
	Color ambientLight = computeAmbientLight( lightSourceList, ambientStrength );
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


ScopedPtr<LightSourceSet> makeGpuLightScopedPtr( const std::list<LightSource> & lightSourceList, float ambientStrength )
{
	Color ambientLight = computeAmbientLight( lightSourceList, ambientStrength );
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
											  const Color & color, float shininess )
{
	TriangleMesh * mesh = new TriangleMesh( triangleList.size(), color, shininess );
	mesh->vertices = new Vector3f[vertexList.size()];
	std::copy( vertexList.begin(), vertexList.end(), mesh->vertices );

	mesh->triangles = new Triangle[mesh->triangleCount];
	std::copy( triangleList.begin(), triangleList.end(), mesh->triangles );

	CleanupCommand * cleanupCommand = new CompositeCleanupCommand(
			{
					new HostCleanupCommand<Triangle>( mesh->triangles, true ),
					new HostCleanupCommand<Vector3f>( mesh->vertices, true ),
					new HostCleanupCommand<TriangleMesh>( mesh, false )
			}
	);

	return ScopedPtr<TriangleMesh>( mesh, cleanupCommand );
}


ScopedPtr<TriangleMesh> makeGpuMeshScopedPtr( const std::list<Vector3f> & vertexList,
											  const std::list<Triangle> & triangleList,
											  const Color & color, float shininess )
{
	TriangleMesh tmpMesh( triangleList.size(), color, shininess );

	checkCudaErrors( cudaMalloc( (void **) &tmpMesh.vertices, vertexList.size() * sizeof( Vector3f ) ) );
	thrust::device_vector<Vector3f> dv( vertexList.begin(), vertexList.end() );
	thrust::copy( dv.begin(), dv.end(), tmpMesh.vertices );

	checkCudaErrors( cudaMalloc( (void **) &tmpMesh.triangles, triangleList.size() * sizeof( Triangle ) ) );
	thrust::device_vector<Triangle> tv( triangleList.begin(), triangleList.end() );
	thrust::copy( tv.begin(), tv.end(), tmpMesh.triangles );

	TriangleMesh * mesh = nullptr;
	checkCudaErrors( cudaMalloc( (void **) &mesh, sizeof( TriangleMesh ) ) );
	checkCudaErrors( cudaMemcpy( mesh, &tmpMesh, sizeof( TriangleMesh ), cudaMemcpyHostToDevice ) );

	CleanupCommand * cleanupCommand = new CompositeCleanupCommand(
			{
					new GpuCleanupCommand<Triangle>( tmpMesh.triangles ),
					new GpuCleanupCommand<Vector3f>( tmpMesh.vertices ),
					new GpuCleanupCommand<TriangleMesh>( mesh )
			}
	);

	return ScopedPtr<TriangleMesh>( mesh, cleanupCommand );
}


#endif //TRIANGLE_RAYCASTING_SCOPEDPTR_H
