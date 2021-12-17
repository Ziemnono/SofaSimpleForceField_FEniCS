# Simple implementation of a Saint-Venant-Kirchhoff force field

This plugin can be used as a simple demonstration of how SOFA 
"force fields" components are built up. In this case, the force
field API will provide the internal potential energy,
residual and jacobian of a Saint-Venant-Kirchhoff material assembled
on linear tetrahedral meshes. These three function will be used 
by the system assembler and ODE solvers of SOFA.

![example workflow](https://github.com/Ziemnono/SofaSimpleForceField_FEniCS//actions/workflows/ubuntu-v21-06-02.yml/badge.svg)
![example workflow](https://github.com/Ziemnono/SofaSimpleForceField_FEniCS//actions/workflows/ubuntu-main.yml/badge.svg)

### To use FEniCS
```console
docker run -ti -v $(pwd):/root dolfinx/dolfinx:latest
```

## Compiling with Ubuntu LTS 20.04
### SOFA (skip this if you already compiled SOFA)

```console
sudo apt install qtbase5-dev libqt5charts5-dev libqt5opengl5-dev libopengl0 libeigen3-dev libglew-dev zlib1g-dev libboost-dev libboost-filesystem-dev g++ cmake git
export SOFA_SRC=/opt/sofa_src
export SOFA_ROOT=/opt/sofa
git clone https://github.com/sofa-framework/sofa.git $SOFA_SRC
cmake -DCMAKE_INSTALL_PREFIX=$SOFA_ROOT -DCMAKE_BUILD_TYPE=Release -S $SOFA_SRC -B $SOFA_SRC/build
cmake --build $SOFA_SRC/build -j4
cmake --install $SOFA_SRC/build
```
### Caribou (if we decide to keep it as a dependency)

### SofaSimpleForcefield
```console
export SSFF_SRC=/opt/SofaSimpleForceField_FEniCS
git clone git@github.com:Ziemnono/SofaSimpleForceField_FEniCS.git $SSFF_SRC
cmake -DCMAKE_PREFIX_PATH="$SOFA_ROOT/lib/cmake,$CARIBOU_ROOT/lib/cmake" -DCMAKE_INSTALL_PREFIX=$SOFA_ROOT/plugins/SofaSimpleForceField -DCMAKE_BUILD_TYPE=Release -S $SSFF_SRC -B $SSFF_SRC/build
cmake --build $SSFF_SRC/build -j4
cmake --install $SSFF_SRC/build
```

## Running the cantilever example scene
```console
$SOFA_ROOT/bin/runSofa -l SofaSimpleForceField $SSFF_SRC/cantilever_beam.scn
```

## Result
![image](https://user-images.githubusercontent.com/6951981/127413110-76fb452e-723b-4e74-b3ac-5952d54d663d.png)

## SOFA Week 2021
Slides: https://docs.google.com/presentation/d/1IilG7QVBciP1qHR0bZ5opFlZvQbmStXUiH2ERznvnoE/edit?usp=sharing

Presentation recording: https://youtu.be/xxoUlOtAkWk?t=7735
