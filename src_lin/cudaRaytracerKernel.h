#pragma once

#include "glm/glm.hpp"
#include "util.h"

__global__ void raycast(unsigned char* const outputImage, int width, int height, 
                        const _Sphere* const spheres,  const _Triangle* const triangles, 
                        const _Light* const lights, _Material* mtl, _CameraData cameraSetting );