//
// Created by bartosz on 1/9/21.
//

#pragma once

#include "TriangleMesh.h"

// utility function for reading some primitives from text files;
// used for loading lights

static Color readColor( const std::string & line )
{
	int space1 = line.find( ' ' );
	if ( space1 == std::string::npos )
		throw std::runtime_error( "Invalid line" );
	std::string rStr = line.substr( 0, space1 );
	float r = std::stof( rStr );

	int space2 = line.find( ' ', space1 + 1 );
	if ( space2 == std::string::npos )
		throw std::runtime_error( "Invalid line" );
	std::string gStr = line.substr( space1 + 1, space2 );
	float g = std::stof( gStr );

	std::string bStr = line.substr( space2 + 1 );
	float b = std::stof( bStr );

	return Color( r, g, b );
}


static Triangle readTriangle( const std::string & line )
{
	int space1 = line.find( ' ' );
	if ( space1 == std::string::npos )
		throw std::runtime_error( "Invalid line" );
	std::string aStr = line.substr( 0, space1 );
	int a = std::stoi( aStr );

	int space2 = line.find( ' ', space1 + 1 );
	if ( space2 == std::string::npos )
		throw std::runtime_error( "Invalid line" );
	std::string bStr = line.substr( space1 + 1, space2 );
	int b = std::stoi( bStr );

	std::string cStr = line.substr( space2 + 1 );
	int c = std::stoi( cStr );

	return { a, b, c };
}


static Vector3f readVertex( const std::string & line )
{
	int space1 = line.find( ' ' );
	if ( space1 == std::string::npos )
		throw std::runtime_error( "Invalid line" );
	std::string xStr = line.substr( 0, space1 );
	float x = std::stof( xStr );

	int space2 = line.find( ' ', space1 + 1 );
	if ( space2 == std::string::npos )
		throw std::runtime_error( "Invalid line" );
	std::string yStr = line.substr( space1 + 1, space2 );
	float y = std::stof( yStr );

	std::string zStr = line.substr( space2 + 1 );
	float z = std::stof( zStr );

	return Vector3f( x, y, z );
}