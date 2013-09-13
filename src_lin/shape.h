#pragma once
#define GLM_SWIZZLE
#include "glm/glm.hpp"

class Shape
{
public:
    Shape(void){}
    virtual ~Shape(void){}

    virtual bool testRayIntersection( const glm::vec3 &raysource, const glm::vec3 &raydir, float &distance ) const = 0;

    virtual glm::vec3 getColor( const glm::vec3 &iDir, const glm::vec3 point ) const = 0;

    virtual glm::vec3 getNormalInPoint( const glm::vec3 &point ) const = 0;

    virtual std::string toString()const{}

    glm::mat4 transform;
    glm::mat4 invTrans;
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 emission;
    glm::vec3 ambient;
    float shininess;
};

