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
    void cleanUp();
    void packSceneDescData( const SceneDesc &sceneDesc );

    //Host-side and packed data for transferring to the device
    _CameraData cameraData;
    _Primitive* h_pPrimitives;
    _Light* h_pLights;
    _Material* h_pMaterials;

    int numPrimitive;
    int numTriangle;
    int numSphere;
    int numLight;
    int numMaterial;

    unsigned char* d_outputImage;
    unsigned char* h_outputImage;
    _Primitive* d_primitives;
    _Light* d_lights;
    _Material* d_materials;

};