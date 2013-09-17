#include <helper_math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cudaRaytracerKernel.h"

__device__ glm::vec3 getSurfaceNormal( glm::vec3* point, const _Primitive* const primitive )
{
    if( primitive->type == 0 ) //sphere
        return glm::normalize( *point - primitive->center );
    else
        return primitive->pn;
}

__device__ float raySphereIntersect( const _Primitive* const sphere, 
                                     const glm::vec3* const raySource, const glm::vec3* const rayDir
                                      )
{
   glm::vec3 dst = *raySource - sphere->center;
   float B = glm::dot( dst, *rayDir );
   float C = glm::dot( dst, dst ) - sphere->radius * sphere->radius;
   float D = B*B - C;
   
   return D > 0 ? ( -B-sqrt(D) > 0 ? -B-sqrt(D) : ( -B+sqrt(D) > 0 ? -B+sqrt(D) : FLOAT_INF) ) : FLOAT_INF;
    
}

__device__ float rayTriangleIntersect( const _Primitive* const triangle, const glm::vec3* const raySource, const glm::vec3* const rayDir )
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

    if( glm::dot( triangle->pn, *rayDir ) == 0 ) //the ray and the plane are parallel
        return FLOAT_INF;

    ray_offset = ( plane_delta - glm::dot( triangle->pn, *raySource ) ) /
                    glm::dot( triangle->pn, *rayDir ) ;

    point = *raySource + ( ray_offset * (*rayDir) );

    BAcrossQA = glm::cross( triangle->vert[1] - triangle->vert[0], point - triangle->vert[0] );
    CBcrossQB = glm::cross( triangle->vert[2] - triangle->vert[1], point - triangle->vert[1] );
    ACcrossQC = glm::cross( triangle->vert[0] - triangle->vert[2], point - triangle->vert[2] );

 
    if( ray_offset > 0 && glm::dot( BAcrossQA, triangle->pn ) >= 0 &&
          glm::dot( CBcrossQB, triangle->pn ) >= 0 &&
        glm::dot( ACcrossQC, triangle->pn ) >= 0 )   
    {
      
        return ray_offset;
    }
    else
        return FLOAT_INF;
}


__device__ glm::vec3 shade( glm::vec3* point, glm::vec3* normal, glm::vec3* eyeRay, const _Primitive* const primitive, 
                           const _Material* const mtl, const _Light* const light )
{
    glm::vec3 L;
    glm::vec3 H;
    float lightDst; //Distance of light source
    float attenu;   //attenuation factor
    //glm::vec3 color = primitive->emission + primitive->ambient;
    glm::vec3 color(0.0f,0.0f,0.0f);

    if( light->pos[3] > 0 ) //local light
    {
        L = glm::normalize( glm::vec3(light->pos) - (*point) );
        //lightDst = glm::distance( (*point), glm::vec3(light->pos));
        lightDst = glm::length( L );
        //attenu = light->attenu_const + 
        //            ( light->attenu_linear + light->attenu_quadratic * lightDst ) * lightDst;
    }
    else
    {
        L = glm::normalize( glm::vec3(light->pos) );
        attenu = 1.0f;
    }

    if( glm::dot( L, *normal ) <= 0 ) //the face is turned away from this light
        return color;

    H = glm::normalize( L - *eyeRay );
    color = light->color   *
                ( primitive->diffuse * fmaxf( glm::dot( *normal, L ), 0.0f ) +
                primitive->specular * powf( fmaxf( glm::dot( *normal, H ), 0.0f ), primitive->shininess ) );


    return color;
}

__device__ int raytrace( const glm::vec3* const ray, const glm::vec3* const source,
                         const _Primitive* const primitives, int primitiveNum,
                         const _Light* const lights, int lightNum, glm::vec3* point, glm::vec3* surfaceNormal )
{
    float nearest = FLOAT_INF;
    float dst;
    int   id = -1;
    glm::vec3 rayLocalCoord;
    glm::vec3 sourceLocalCoord;
    glm::vec3 tmpP, tmpN;

    //__shared__ h_Primitive; 

    for( int i = 0; i < primitiveNum; ++i )
    {
        //transform ray to object coordinate
        rayLocalCoord =  glm::normalize( glm::vec3( primitives[i].invTrans * glm::vec4( *ray, 0.0f ) ) );
        sourceLocalCoord = glm::vec3( primitives[i].invTrans * glm::vec4( *source, 1.0f ) );

        if( primitives[i].type == 0 ) //sphere
        {
            dst = raySphereIntersect( primitives+i, &sourceLocalCoord ,&rayLocalCoord );
        }
        else
        {
            dst = rayTriangleIntersect( primitives+i,  &sourceLocalCoord, &rayLocalCoord );

        }
        if( FLOAT_INF == dst )
           continue;

        tmpP = sourceLocalCoord + ( dst * rayLocalCoord );

        //transform the incident point and normal to world coordinate
        
        tmpN = getSurfaceNormal( &tmpP, primitives+i );
        tmpN = glm::normalize( glm::transpose( glm::mat3( primitives[i].invTrans ) ) * (tmpN) );
        tmpP = glm::vec3( (primitives[i].transform * glm::vec4( tmpP, 1.0f )) );

        if( glm::dot( tmpN, *ray ) > 0 ) //surface turnes away from the camera
            continue;
      
        dst = glm::distance( tmpP, *source );

        if( dst < nearest )
        {
            nearest = dst;
            id = i;
            *point = tmpP;
            *surfaceNormal = tmpN;
        }
    }

     return id;
}


__device__ float shadowTest( glm::vec3* point, glm::vec3* normal, const _Primitive* const occluders, int occluderNum,
                             const _Light* const light )
{
    glm::vec3 L, LLocalFrame;
    glm::vec3 PLocalFrame;
    glm::vec3 O;
    float lightDst, occluderDst;
    float shadowPct = 0;
    float delta;
    ushort2 LSample;
  

    if( light->type == 0 ) //point local light
    {    
        //lightDst = glm::distance( *point, glm::vec3(light->pos) ); //distance in world coord
        L = glm::vec3(light->pos) - *point ;
        lightDst = glm::length( L );
        LSample.x = LSample.y = 1;
    }
    else if( light->type == 1 ) //point directional light
    {    
        lightDst = FLOAT_INF;
        L = glm::vec3(light->pos);
        LSample.x = LSample.y = 1;
    }
    else if( light->type == 2 ) //area light
    {
        //lightDst = glm::distance( *point, glm::vec3(light->pos) ); //distance in world coord
        L = glm::vec3(light->pos) - *point ;
        lightDst = glm::length( L );
        LSample.x = LSample.y = 4;
    }

    if( glm::dot( *normal, L ) < 0 ) 
        return 1.0f;

    delta = 1.0f/(LSample.x*LSample.y );

    for( int y = 0; y < LSample.y; ++y ) for( int x = 0; x < LSample.x; ++x )
    {
        for( int i = 0; i < occluderNum; ++i )
        {
            //transform the light vector to object space and normalize it
            LLocalFrame = glm::normalize( glm::mat3( occluders[i].invTrans ) * L ); 

            //transform the test point to object space
            PLocalFrame = glm::vec3( occluders[i].invTrans * glm::vec4( *point, 1.0f ) );

            if( occluders[i].type == 0 ) //sphere
            {
                occluderDst = raySphereIntersect( occluders+i, &PLocalFrame ,&LLocalFrame );
            }
            else
            {
                occluderDst = rayTriangleIntersect( occluders+i,  &PLocalFrame, &LLocalFrame );

            }
            if( FLOAT_INF == occluderDst )
               continue;

            //transform the occluder point to world frame
            O = glm::vec3( occluders[i].transform * 
                            glm::vec4( PLocalFrame + occluderDst *  LLocalFrame, 1 ) );

            occluderDst = glm::distance( *point,  O );
            if( occluderDst < lightDst )
            {
                shadowPct += delta;
                //return shadowPct;
            }
        
        }
        L = ( glm::vec3(light->pos) + glm::vec3( light->width * ( (1.0f+x)/(float)LSample.x ), light->width * ( (1.0f+y)/(float)LSample.y ), 0.0f ) ) -
               *point;
        lightDst = glm::length( L );

    }
    return shadowPct;
}

__global__ void raycast( unsigned char* const outputImage, int width, int height, _CameraData cameraData,
                         const _Primitive* const primitives, int primitiveNum,
                         const _Light* const lights, int lightNum, _Material* mtl, int mtlNum )
{
    
    ushort2 idx;
    float2 offset;
    glm::vec3 ray;
    glm::vec3 raysource;
    glm::vec3 incidentP;
    glm::vec3 shiftP;
    glm::vec3 surfaceNormal;
    glm::vec3 color(0.0f,0.0f,0.0f);
    glm::vec3 finalColor(0.0f,0.0f,0.0f);
    glm::vec3 cumulativeSpecular( 1.0f, 1.0f, 1.0f );
    int hitId;
    float shadowPct;
   
    int outIdx;

    //generate ray based on block and thread idx
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if( idx.x > width || idx.y > height )
        return;

    outIdx = idx.y * width * 3 + 3 * idx.x; //element to shade in the output buffer

    offset.x = cameraData.viewportHalfDim.x * ( (idx.x+0.5) / (width/2.0) - 1 );
    offset.y = cameraData.viewportHalfDim.y * ( 1- (idx.y+0.5) / (height/2.0)  );

    ray = cameraData.wVec + offset.x * cameraData.vVec + offset.y * cameraData.uVec;
    ray = glm::normalize( ray );
    raysource = cameraData.eyePos;

    for( int depth = 0; depth < 5; ++depth )
    {
        color.x = color.y = color.z = 0.0f; //clear color vector for use in current iteration
        hitId = raytrace( &ray,&raysource, primitives, primitiveNum, lights, lightNum, &incidentP, &surfaceNormal );
        if( hitId >= 0 )
        {
            shiftP = incidentP +  (0.001f * surfaceNormal);
            for( int i = 0; i < lightNum; ++i )
            {
                shadowPct = shadowTest( &shiftP, &surfaceNormal, primitives, primitiveNum, lights+i );

                //sahding
                //if( shadowPct ==0 )
                  color += (1.0f-shadowPct)*shade( &incidentP, &surfaceNormal, &ray, &primitives[hitId], 0, lights+i );
            }
            color += primitives[hitId].ambient + primitives[hitId].emission;
    
        
        }
        finalColor += color * cumulativeSpecular;
        
        if( glm::all(glm::equal(primitives[hitId].specular, glm::vec3(0.0f,0.0f,0.0f) ) ) )
            break;

        ray = glm::normalize( glm::reflect( ray, surfaceNormal ) );
        raysource = shiftP;
        cumulativeSpecular *= primitives[hitId].specular;
    }
    //write color to output buffer
    outputImage[ outIdx ] = finalColor.x > 1 ? 255 : finalColor.x * 255;
    outputImage[ outIdx + 1] = finalColor.y > 1 ? 255 : finalColor.y * 255;
    outputImage[ outIdx + 2] = finalColor.z > 1 ? 255 : finalColor.z * 255 ;

}

void rayTracerKernelWrapper( unsigned char* const outputImage, int width, int height, _CameraData cameraData,
                              const _Primitive* const primitives, int primitiveNum,
                              const _Light* const lights, int lightNum, _Material* mtl, int mtlNum )
{
    dim3 blockSize = dim3( 32, 32 );
    dim3 gridSize = dim3( (width + blockSize.x-1)/blockSize.x, (height + blockSize.y-1)/blockSize.y );

    //The ray tracing work is done in the kernel
    raycast<<< gridSize, blockSize >>>( outputImage, width, height, cameraData, primitives, primitiveNum,
                                       lights, lightNum, mtl, mtlNum );
    cudaErrorCheck( cudaGetLastError() );
    cudaDeviceSynchronize(); 
    cudaErrorCheck( cudaGetLastError() );
}