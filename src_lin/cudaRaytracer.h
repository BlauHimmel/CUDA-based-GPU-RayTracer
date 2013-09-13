#pragma once

#include "raytracer.h"
#include "sceneDesc.h"
#include "ColorImage.h"
#include "util.h"

class CudaRayTracer: public RayTracer
{
public:
    CudaRayTracer();
    ~CudaRayTracer();
    void renderImage(const SceneDesc &scene, ColorImage &img);
private:
    void clearData();
    void packSceneDescData( const SceneDesc &sceneDesc );

    _CameraData cameraData;
    _Triangle* pTriangles;
    _Sphere* pSpheres;
    _Material* pMaterials;

};