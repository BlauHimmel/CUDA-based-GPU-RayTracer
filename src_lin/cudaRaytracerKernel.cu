#include <helper_math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "glm/glm.hpp"
#include "cudaRaytracerKernel.h"

__device__ float raySphereIntersect( const _Sphere* const sphere, glm::vec3 raySource, glm::vec3 rayDir )
{
   glm::vec3 dst = raySource - sphere->center;
   float B = glm::dot( dst, rayDir );
   float C = glm::dot( dst, dst ) - sphere->radius;
   float D = B*B - C;
   return D > 0 ? ( -B-sqrt(D) ) : FLOAT_INF;
    
}

__device__ float rayTriangleIntersect( const _Triangle* const triangle, glm::vec3 raySource, glm::vec3 rayDir )
{
        //glm::vec3 planeNormal;
    glm::vec3 BAcrossQA;
    glm::vec3 CBcrossQB;
    glm::vec3 ACcrossQC;
    glm::vec3 point;

    float plane_delta;
    float ray_offset;

    //planeNormal = glm::normalize( glm::cross( v[1] - v[0], v[2] - v[0] ) );
    plane_delta = glm::dot( triangle->pn, triangle->vert[0] );

    if( glm::dot( triangle->pn, rayDir ) == 0 ) //the ray and the plane are parallel
        return FLOAT_INF;

    ray_offset = ( plane_delta - glm::dot( triangle->pn, raySource ) ) /
                    glm::dot( triangle->pn, rayDir ) ;

    point = raySource + ( ray_offset * rayDir );

    BAcrossQA = glm::cross( triangle->vert[1] - triangle->vert[0], point - triangle->vert[0] );
    CBcrossQB = glm::cross( triangle->vert[2] - triangle->vert[1], point - triangle->vert[1] );
    ACcrossQC = glm::cross( triangle->vert[0] - triangle->vert[2], point - triangle->vert[2] );

    if( ray_offset < 0 )
        return FLOAT_INF;
 
    else if( glm::dot( BAcrossQA, triangle->pn ) >= 0 &&
          glm::dot( CBcrossQB, triangle->pn ) >= 0 &&
        glm::dot( ACcrossQC, triangle->pn ) >= 0 )   
    {
      
        return ray_offset;
    }
    else
        return FLOAT_INF;
}

__device__ bool raytrace( glm::vec3 ray, glm::vec3 source, const _Sphere* const spheres, int sphereNum,
                         const _Triangle* const triangles, int triangleNum,
                         const _Light* const lights, int lightNum )
{
    float nearest = FLOAT_INF;  //nearest distance
    float dst;
    glm::vec3 incidentPoint;

    for( int i = 0; i < sphereNum; ++i )
    {
        dst = raySphereIntersect( &spheres[i], source, ray );
        if( dst > nearest ) 
            continue;
        else
            nearest = dst;
    }

}

__global__ void raycast( unsigned char* const outputImage, int width, int height, _CameraData cameraSetting,
                         const _Sphere* const spheres, int sphereNum, const _Triangle* const triangles, int triangleNum,
                         const _Light* const lights, int lightNum, _Material* mtl, int mtlNum )
{
    
    short2 idx;
    float2 offset;
    glm::vec3 ray;

    //generate ray based on block and thread idx
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    offset.x = cameraSetting.viewportHalfDim.x * ( (idx.x+0.5) / (width/2.0) - 1 );
    offset.y = cameraSetting.viewportHalfDim.y * ( (idx.y+0.5) / (height/2.0) - 1 );

    ray = cameraSetting.wVec + offset.x * cameraSetting.vVec + offset.y * cameraSetting.uVec;
    ray = glm::normalize( ray );
    //shadow ray test

    //shading
    if( raytrace( ray,cameraSetting.eyePos, spheres, sphereNum, triangles, triangleNum, lights, lightNum,  ) )
    {
    }

    //write color to output buffer
}