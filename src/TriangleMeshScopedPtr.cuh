//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_TRIANGLEMESHSCOPEDPTR_CUH
#define TRIANGLE_RAYCASTING_TRIANGLEMESHSCOPEDPTR_CUH

#include "CleanupCommand.cuh"
#include "dependencies/helper_cuda.h"
#include "TriangleMesh.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <list>


class TriangleMeshScopedPtr
{
protected:
	TriangleMesh * mesh = nullptr;
	CleanupCommand * cleanupCommand = nullptr;

public:
	TriangleMeshScopedPtr( TriangleMesh * mesh, CleanupCommand * cleanupCommand )
	{
		this->mesh = mesh;
		this->cleanupCommand = cleanupCommand;
	}

	TriangleMesh * operator->()
	{
		return mesh;
	}

	TriangleMesh * getMesh() const
	{
		return mesh;
	}

	~TriangleMeshScopedPtr()
	{
		cleanupCommand->execute();
		delete cleanupCommand;
	}
};


TriangleMeshScopedPtr makeCpuMeshScopedPtr( const std::list<Vector3f> & vertexList,
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
					new HostCleanupCommand<Color>( mesh->colors ),
					new HostCleanupCommand<Triangle>( mesh->triangles ),
					new HostCleanupCommand<Vector3f>( mesh->vertices ),
					new HostCleanupCommand<TriangleMesh>( mesh )
			}
	);

	return TriangleMeshScopedPtr( mesh, cleanupCommand );
}

TriangleMeshScopedPtr makeGpuMeshScopedPtr( const std::list<Vector3f> & vertexList,
											const std::list<Triangle> & triangleList,
											const std::list<Color> & colorList )
{
	TriangleMesh * tmpMesh = new TriangleMesh( triangleList.size() );

	checkCudaErrors( cudaMalloc( (void **) &(tmpMesh->vertices), vertexList.size() * sizeof( Vector3f ) ) );
	thrust::device_vector<Vector3f> dv(vertexList.begin(), vertexList.end());
	thrust::copy( dv.begin(), dv.end(), tmpMesh->vertices );

	checkCudaErrors( cudaMalloc( (void **) &(tmpMesh->triangles), triangleList.size() * sizeof( Triangle ) ) );
	thrust::device_vector<Triangle> tv(triangleList.begin(), triangleList.end());
	thrust::copy( tv.begin(), tv.end(), tmpMesh->triangles );

	checkCudaErrors( cudaMalloc( (void **) &(tmpMesh->colors), colorList.size() * sizeof( Color ) ) );
	thrust::device_vector<Color> cv(colorList.begin(), colorList.end());
	thrust::copy( cv.begin(), cv.end(), tmpMesh->colors );

	TriangleMesh * mesh = nullptr;
	checkCudaErrors( cudaMalloc( (void **) &mesh, sizeof( TriangleMesh ) ) );
	checkCudaErrors( cudaMemcpy( mesh, tmpMesh, sizeof( TriangleMesh ), cudaMemcpyHostToDevice ) );

	CleanupCommand * cleanupCommand = new CompositeCleanupCommand(
			{
					new GpuCleanupCommand<Color>( tmpMesh->colors ),
					new GpuCleanupCommand<Triangle>( tmpMesh->triangles ),
					new GpuCleanupCommand<Vector3f>( tmpMesh->vertices ),
					new GpuCleanupCommand<TriangleMesh>( mesh )
			}
	);

	delete tmpMesh;
	return TriangleMeshScopedPtr( mesh, cleanupCommand );
}


#endif //TRIANGLE_RAYCASTING_TRIANGLEMESHSCOPEDPTR_CUH
