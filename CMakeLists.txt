cmake_minimum_required(VERSION 3.15)
project(SofaSimpleForcefield VERSION 1.0)

find_package(Sofa.Core REQUIRED)
find_package(SofaCaribou REQUIRED)
find_package(Caribou REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fenics)

add_library(${PROJECT_NAME} SHARED src/init.cpp
    src/SVKElasticForcefield_SOFA.cpp
    src/SVKElasticForcefield_FEniCS.cpp
    fenics/SaintVenantKirchhoff_Tetra.c
    fenics/SaintVenantKirchhoff_Hexa.c
    src/SVKElasticForcefield_SOFA_Tetra_Order2.cpp
    src/SVKElasticForcefield_SOFA_Hexa.cpp
    src/SVKElasticForcefield_FEniCS_Hexa.cpp
    #src/Python/SVKElasticForcefield_FEniCS.cpp
    )
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Core Eigen3::Eigen Caribou::Config Caribou::Algebra Caribou::Geometry Caribou::Topology Caribou::Mechanics SofaCaribou)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
)
