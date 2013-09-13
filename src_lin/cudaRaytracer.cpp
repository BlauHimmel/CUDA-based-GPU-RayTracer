#include "cudaRaytracer.h"
#include "cudaRaytracerKernel.h"

CudaRayTracer::CudaRayTracer()
{
    pSpheres = 0;
    pTriangles = 0;
    pMaterials = 0;
}

CudaRayTracer::~CudaRayTracer()
{
    clearData();
}

void CudaRayTracer::renderImage( const SceneDesc &scene, ColorImage &img)
{
    if( scene.width < 1 || scene.height < 1 )
        return;
    //Pack scene description data


    //allocate image in the device


    //Send scene description data to the device

    //raycasting, performed in the cuda kernel

    //Read back the image

}

void CudaRayTracer::packSceneDescData( const SceneDesc &sceneDesc )
{
    int sphereNum = 0;
    int triangleNum = 0;

    //counting the number of spheres and triangles 
    for( int i = 0; i < sceneDesc.primitives.size(); ++i )
    {
        if( sceneDesc.primitives[i]->toString().compare("sphere") == 0 )
            sphereNum++;
        else if( sceneDesc.primitives[i]->toString().compare("triangle") == 0 )
            triangleNum++;
    }

    //packing the camrea setting
    cameraData.eyePos = sceneDesc.eyePos;
    cameraData.viewportHalfDim.y = tan( sceneDesc.fovy / 2.0 );
    cameraData.viewportHalfDim.x = (float)sceneDesc.width / (float) sceneDesc.height * cameraData.viewportHalfDim.y;

    //construct the 3 orthoogonal vectors constitute a frame
    cameraData.uVec = glm::normalize( sceneDesc.up );
    cameraData.wVec = glm::normalize( sceneDesc.center - sceneDesc.eyePos );
    cameraData.vVec = glm::normalize( glm::cross( cameraData.wVec, cameraData.uVec ) );
    cameraData.uVec = glm::normalize( glm::cross( cameraData.vVec, cameraData.wVec ) );

    clearData();

    //packing the primitives
    pSpheres = new _Sphere[sphereNum];
    pTriangles = new _Triangle[triangleNum];

    int sCount = 0; 
    int tCount = 0;
    for( int i = 0; i < sceneDesc.primitives.size(); ++i )
    {
        if( sceneDesc.primitives[i]->toString().compare("sphere") == 0 )
        {
            pSpheres[sCount].center = ((Sphere*)sceneDesc.primitives[i])->center.xyz;
            pSpheres[sCount].radius = ((Sphere*)sceneDesc.primitives[i])->radius;
            pSpheres[sCount].transform = sceneDesc.primitives[i]->transform;
            pSpheres[sCount].invTrans = sceneDesc.primitives[i]->invTrans;
            ++sCount;
        }
        else if( sceneDesc.primitives[i]->toString().compare("triangle") == 0 )
        {
            for( int n = 0; n < 3; n++ )
                pTriangles[tCount].vert[n]  =((Triangle*)sceneDesc.primitives[i])->v[n];

            for( int n = 0; n < 3; n++ )
                pTriangles[tCount].normal[n] =((Triangle*)sceneDesc.primitives[i])->n[n];

            pTriangles[tCount].transform = sceneDesc.primitives[i]->transform;
            pTriangles[tCount].invTrans = sceneDesc.primitives[i]->invTrans;
            ++tCount;
        }
    }
}

void CudaRayTracer::clearData()
{
    if( pSpheres )
        delete [] pSpheres;
    pSpheres = 0;

    if( pTriangles )
        delete [] pTriangles;
    pTriangles = 0;

    if( pMaterials )
        delete [] pMaterials;
    pMaterials = 0;
}