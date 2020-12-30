cmake_minimum_required(VERSION 3.17)
project(triangle_raycasting CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(triangle_raycasting src/main.cu src/RayCaster.h src/utils.cu src/utils.cuh src/Vector3f.cuh src/TriangleMesh.h src/CpuRayCaster.h src/MeshLoader.h src/IndexMeshLoader.h src/Renderer.h src/GlBasicRenderer.h src/Camera.h)
target_link_libraries(triangle_raycasting -lglut -lGL -lGLU glfw -lGLEW)

set_target_properties(
        triangle_raycasting
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)