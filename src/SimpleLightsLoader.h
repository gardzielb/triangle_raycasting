//
// Created by bartosz on 1/9/21.
//

#pragma once

#include <list>
#include "LightsLoader.h"
#include "readFileUtils.h"


class SimpleLightsLoader : public LightsLoader
{
private:
	std::string fileName;

public:
	SimpleLightsLoader( const std::string & fileName )
			: fileName( fileName )
	{}

	ScopedPtr<LightSourceSet> loadLights( DestMemoryKind memoryKind ) const override
	{
		std::list<LightSource> lightSourceList;
		std::ifstream stream( fileName );
		std::string line;

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
			return makeCpuLightScopedPtr( lightSourceList );
		return makeGpuLightScopedPtr( lightSourceList );
	}
};
