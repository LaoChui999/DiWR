cmake_minimum_required(VERSION 3.18)

project(DWG_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories must be different.")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CGAL REQUIRED COMPONENTS Core)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
find_package(PCL REQUIRED COMPONENTS common kdtree search features)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/KNN_GWN.cu
    src/RMSProp.cu
    src/TurboDWG.cu
    src/TurboLibiGWN.cu
    src/TurboMC.cu
    src/TurboNormalInit.cu
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cudaKDTree
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include
)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE
    ${PCL_DEFINITIONS}
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/MP;/bigobj>
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS NOMINMAX _USE_MATH_DEFINES
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
    )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    CGAL::CGAL
    CGAL::CGAL_Core
    OpenMP::OpenMP_CXX
    Threads::Threads
    ${PCL_LIBRARIES}
)
