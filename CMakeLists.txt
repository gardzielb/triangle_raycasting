cmake_minimum_required(VERSION 3.17)
project(triangle_raycasting CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

add_executable(triangle_raycasting src/main.cu src/RayCaster.h src/raycasting.cu src/raycasting.cuh src/Vector3f.h src/TriangleMesh.h src/CpuRayCaster.h src/MeshLoader.h src/Renderer.h src/GlBasicRenderer.h src/Camera.h src/GpuRayCaster.h src/ScopedPtr.h src/CleanupCommand.h src/kernels.cu src/kernels.cuh src/Matrix3f.h src/Color.h src/LightsLoader.h src/SimpleLightsLoader.h src/readFileUtils.h src/ObjMeshLoader.h)
target_link_libraries(triangle_raycasting -lglut -lGL -lGLU glfw -lGLEW -lassimp)

set_target_properties(
        triangle_raycasting
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)