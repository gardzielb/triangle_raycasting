//
// Created by bartosz on 1/9/21.
//

#pragma once

#include <list>
#include "LightsLoader.h"
#include "readFileUtils.h"


// loads lights data from text file
class SimpleLightsLoader : public LightsLoader
{
private:
	std::string fileName;

public:
	SimpleLightsLoader( const std::string & fileName )
			: fileName( fileName )
	{}

	// expects file with ambientStrength in the first line and light sources data in the following lines
	// light source data format: <position> s <specular color> d <diffuse color> \n
	ScopedPtr<LightSourceSet> loadLights( DestMemoryKind memoryKind ) const override
	{
		std::list<LightSource> lightSourceList;
		std::ifstream stream( fileName );
		std::string line;

		getline( stream, line );
		float ambientStrength = std::stof( line );

		while ( getline( stream, line ) )
		{
			int sPos = line.find( 's' );
			int dPos = line.find( 'd' );
			std::string vertexStr = line.substr( 0, sPos - 1 );
			std::string specularStr = line.substr( sPos + 2, dPos - 1 );
			std::string diffusePos = line.substr( dPos + 2 );
			lightSourceList.emplace_back( readVertex( vertexStr ), readColor( specularStr ), readColor( diffusePos ) );
		}
		stream.close();

		if ( memoryKind == DestMemoryKind::CPU )
			return makeCpuLightScopedPtr( lightSourceList, ambientStrength );
		return makeGpuLightScopedPtr( lightSourceList, ambientStrength );
	}
};
