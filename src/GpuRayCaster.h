//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_GPURAYCASTER_H
#define TRIANGLE_RAYCASTING_GPURAYCASTER_H

#include "RayCaster.h"
#include "kernels.cuh"
#include <cuda.h>


// A GPU implementation of a RayCaster
class GpuRayCaster : public RayCaster
{
private:
	// pointers to necessary data copies allocated on GPU
	PaintScene * gpuScene = nullptr;
	Color * gpuPixels = nullptr;
	Camera * gpuCamera = nullptr;

	dim3 threadsPerBlock;
	dim3 blockCount;

public:
	GpuRayCaster( int sceneWidth, int sceneHeight )
	{
		checkCudaErrors( cudaSetDevice( 0 ) );

		int xBlock = std::min( 16, sceneWidth );
		int yBlock = std::min( 16, sceneHeight );
		threadsPerBlock = dim3( xBlock, yBlock );
		blockCount = dim3( ceil( (float) sceneWidth / xBlock ), ceil( (float) sceneHeight / yBlock ) );

		PaintScene tmpScene;
		tmpScene.width = sceneWidth;
		tmpScene.height = sceneHeight;
		checkCudaErrors( cudaMalloc( (void **) &gpuPixels, tmpScene.width * tmpScene.height * sizeof( Color ) ) );
		tmpScene.pixels = gpuPixels;

		gpuScene = nullptr;
		checkCudaErrors( cudaMalloc( (void **) &gpuScene, sizeof( PaintScene ) ) );
		checkCudaErrors( cudaMemcpy( gpuScene, &tmpScene, sizeof( PaintScene ), cudaMemcpyHostToDevice ) );

		checkCudaErrors( cudaMalloc( (void **) &gpuCamera, sizeof( Camera ) ) );
	}

	// calls raycasting kernel to compute pixels values
	void paintTriangleMesh( const ScopedPtr<TriangleMesh> & meshPtr, const ScopedPtr<LightSourceSet> & lightPtr,
							PaintScene & scene, const Camera & camera ) override
	{
		checkCudaErrors( cudaMemcpy( gpuCamera, &camera, sizeof( Camera ), cudaMemcpyHostToDevice ) );

		rayCastingKernel<<<blockCount, threadsPerBlock>>>(
				meshPtr.getObject(), gpuScene, gpuCamera, lightPtr.getObject()
		);

		cudaError_t error = cudaGetLastError();
		if ( error != cudaSuccess )
			throw std::runtime_error( "Error occured during kernel execution: " + std::to_string( (int) error ) );

		checkCudaErrors( cudaDeviceSynchronize() );
		checkCudaErrors( cudaMemcpy(
				scene.pixels, gpuPixels, scene.width * scene.height * sizeof( Color ), cudaMemcpyDeviceToHost
		) );
	}

	~GpuRayCaster()
	{
		cudaFree( gpuCamera );
		cudaFree( gpuScene );
		cudaFree( gpuPixels );
	}
};


#endif //TRIANGLE_RAYCASTING_GPURAYCASTER_H
