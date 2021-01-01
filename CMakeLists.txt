cmake_minimum_required(VERSION 3.17)
project(triangle_raycasting CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

add_executable(triangle_raycasting src/main.cu src/RayCaster.h src/raycasting.cu src/raycasting.cuh src/Vector3f.cuh src/TriangleMesh.h src/CpuRayCaster.h src/MeshLoader.h src/IndexMeshLoader.h src/Renderer.h src/GlBasicRenderer.h src/Camera.cuh src/GpuRayCaster.cuh src/TriangleMeshScopedPtr.cuh src/CleanupCommand.cuh src/kernels.cu src/kernels.cuh src/Matrix3f.cuh src/Color.cuh)
target_link_libraries(triangle_raycasting -lglut -lGL -lGLU glfw -lGLEW)

set_target_properties(
        triangle_raycasting
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)