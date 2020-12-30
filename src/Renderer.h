//
// Created by bartosz on 12/30/20.
//

#ifndef TRIANGLE_RAYCASTING_RENDERER_H
#define TRIANGLE_RAYCASTING_RENDERER_H

#include "TriangleMesh.h"


class Renderer
{
public:
	virtual void renderScene( const PaintScene & scene ) = 0;

	virtual bool isAlive() = 0;

	virtual ~Renderer() = default;
};


#endif //TRIANGLE_RAYCASTING_RENDERER_H
