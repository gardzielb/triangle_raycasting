//
// Created by bartosz on 1/9/21.
//

#pragma once


#include "ScopedPtr.h"
#include "TriangleMesh.h"

// class loading light configuration from file
class LightsLoader
{
public:
	virtual ScopedPtr<LightSourceSet> loadLights( DestMemoryKind memoryKind ) const = 0;

	virtual ~LightsLoader() = default;
};



