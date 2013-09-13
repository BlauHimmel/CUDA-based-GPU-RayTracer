#pragma once

#include "sceneDesc.h"
#include "ColorImage.h"

class RayTracer
{
public: 
    virtual void renderImage( const SceneDesc &scene, ColorImage &img ){};

};