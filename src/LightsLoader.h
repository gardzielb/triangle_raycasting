//
// Created by bartosz on 1/9/21.
//

#pragma once


#include "ScopedPtr.h"
#include "TriangleMesh.h"


class LightsLoader
{
public:
	virtual ScopedPtr<LightSourceSet> loadLights( DestMemoryKind memoryKind ) const = 0;

	virtual ~LightsLoader() = default;
};



