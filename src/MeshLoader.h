//
// Created by bartosz on 12/29/20.
//

#ifndef RAYCASTING_MESHLOADER_H
#define RAYCASTING_MESHLOADER_H

#include <fstream>
#include <list>


class MeshLoader
{
public:
	virtual TriangleMesh loadMesh() = 0;
};

#endif //RAYCASTING_MESHLOADER_H
