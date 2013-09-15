#pragma once
#define GLM_SWIZZLE
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "glm\glm.hpp"

#define FLOAT_INF 0x7F800000

#define cudaErrorCheck( errNo ) checkError( (errNo), __FILE__, __LINE__ )

inline void checkError( cudaError_t err, const char* const filename, const int line  )
{
    if( err != cudaSuccess )
    {
        std::cerr<<"CUDA ERROR: "<<filename<<" line "<<line<<"\n";
        std::cerr<<cudaGetErrorString( err )<<"\n";
        exit(1);
    }
}

//typedef struct _Triangle
//{  
//    glm::mat4 transform;
//    glm::mat4 invTrans;
//
//    //temporary
//    glm::vec3 diffuse;
//    glm::vec3 specular;
//    glm::vec3 emission;
//    glm::vec3 ambient;
//    float shininess;
//
//    //unsigned short material_id;
//
//} _Triangle;

typedef struct _Primitive
{
    int type;

    //used for sphere type
    glm::vec3 center;
    float radius;

    //used for triangle type
    glm::vec3 vert[3];
    glm::vec3 normal[3];
    glm::vec3 pn; //plane normal used when vertex normal not specified

    glm::mat4 transform;
    glm::mat4 invTrans;

    //temporary
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 emission;
    glm::vec3 ambient;
    float shininess;

    //unsigned short material_id;
}_Primitive;

typedef struct _Light
{
    glm::vec4 pos;
    glm::vec3 color;
    float attenu_const;
    float attenu_linear;
    float attenu_quadratic;

}_Light;

typedef struct _CamreaData
{
    glm::vec3 eyePos;
    glm::vec3 uVec;
    glm::vec3 vVec;
    glm::vec3 wVec;
    glm::vec2 viewportHalfDim;

}_CameraData;

typedef struct _Material
{
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 emission;
    glm::vec3 ambient;
    float shininess;
} _Material;